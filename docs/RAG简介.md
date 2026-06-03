---
一、入库阶段（一个视频从音频到"可被搜"的过程）

文件入口：backend/src/bibilab/pipeline/

1. 转录 (transcribe.py)
Whisper 把音频转成一堆带时间戳的小 segment（5–15 秒一段）。
2. 切片 (chunk.py)
贪心合并相邻 segment，凑到 ~300 token 就封一个 RagChunk，超过 400 token 的单段直接独立成块。每块记下 timestamp_start/end 和 sequence_index。
目的：让一块里的内容是一个完整的语义片段，而不是 5 秒一段太碎。
3. 双轨入库 (embed.py: embed_chunks)
每个 chunk 同时写两份索引：

    - 向量库（ChromaDB）：用本地 ONNX MiniLM 模型把文本变成向量，存进 ~/.bibilab/chroma/，metadata 带 video_id / list_id / video_title / 时间戳。
    - 关键词库（SQLite FTS5）：populate_fts() 把同一批 chunk 文本插入 chunks_fts 虚拟表，让 SQLite 的 BM25 能直接打分。

两份都按 video_id 幂等，重新跑会先 delete 老数据再插。

▎ 为什么写两份：向量擅长语义近似（"他在讲什么意思"），BM25 擅长精确字面（"他有没有说过 XX 这个词"）。两边各有失误的场景，所以查询要混搭。

---
二、查询阶段（用户在聊天里问一句的全过程）

文件入口：routers/chat.py → pipeline/chat_tools.py → pipeline/embed.py: retrieve() → pipeline/rerank.py

LLM 在 stream 中自己决定要不要查、查什么、用哪个工具——没有独立的分类器前置调用。检索工具有**两个**，职责不同：

- `find_passages(query, sequence_number?, season_number?)` —— **事实定位器**。在列表里找"和这句话最相关的若干片段"，召回优先。
- `read_source(source_id?, sequence_number?, season_number?)` —— **整篇阅读器**（兜底）。锁定唯一一个来源，返回它的完整连续转录，与排序无关。

LLM 按问题性质选工具：找一个具体事实 → `find_passages`；问"第N集讲了什么 / 某集的剧情" → `read_source`；`find_passages` 发现来源对题但片段没答到点上 → 升级到 `read_source`。`stream_with_tools` 有界循环（最多 3 轮）允许 `find_passages → read_source` 的升级。

### A) find_passages —— 三步管线

1. 确定性 facet 缩池（检索前的唯一收窄手段）

`find_passages` 接 `sequence_number` / `season_number`（整数，可选）。LLM 只负责从问题里**抽取**这两个数（解析，可靠），后端拿它们和 `sources` 表里的列做**精确匹配**（集合运算，确定），在混合检索**之前**把全量来源池缩到匹配子集。

▎ Fail-open 是硬规则：没传 facet → 不筛；传了但零匹配（或 facet 子查询 DB 报错）→ **退回全量来源池，绝不清空**，打 `facet_scope.no_match=true`，并在 LLM 可见的工具结果里加一行事实（"按全部来源搜索"），让模型先声明降级范围再作答。`scoped_pool_size` 记全量池大小，narrowed 数量看 `facet_scope.matched_count`。系统提示是 per-language 静态文本（不内嵌来源清单），Anthropic prompt cache 能覆盖整个前缀。`series_name` 的模糊匹配尚未实现。

2. 混合检索 + RRF 融合 — retrieve() → hybrid_search()

              ┌──> query_chunks (向量)
  query ──┤                              ├──> RRF 融合 ──> 候选池
              └──> query_fts    (BM25)

- 并行两条腿一起跑（asyncio.gather）。
- 候选池大小是动态的：`_dynamic_pool(n) = min(max(n×3, 10), 60)`，再用 `max(pool, top_k)` 兜底——按来源数 3× 伸缩，小列表保底 10，大列表封顶 60（控延迟）。
- RRF 融合（_rrf_fuse，k=60）：每个 chunk 在两个榜里取 1/(k+rank) 累加；无需把"向量距离"和"BM25 分"换算到同一尺度。
- 降级：FTS 出错或空 → 纯向量；向量出错 → 纯 FTS。

3. cross-encoder 精排 → **gateless top-k**

把候选和原 query 拼成 pair，丢给 `Xenova/bge-reranker-base`（XLM-RoBERTa，中英双语；懒加载、单例）打一个"这句和那句相不相关"的分数。重排失败也不会崩，回退到未重排的候选（`reranked=False`）。

然后**直接按精排序取前 `FIND_PASSAGES_TOP_K = 8` 块**——没有相关度门，也没有多样性截断。

▎ 为什么不设门控（gateless）：**精排是排序，不是权威**。两个真实硬例（同音词实体 `前美子是谁`、抽象错配 `隐身术的施法三要素`）里，相关度门会把对的来源/对的块也滤掉，精排分甚至和正确性反相关。所以采用召回优先的纯 top-k，把"够不够、要不要继续读"的判断交给 LLM（grounding prompt 教它：片段能答 → 综合；对题但没答到点 → `read_source`）。

入选的 8 块再用 `transcript_segments` 重建**说话人轮次正文**（`format_turns`，含时间戳）喂给 LLM；精排/向量本身仍用原始 `chunk.content`。最后按来源分组，套 `===== Source [N]: "title" =====` 围栏——同一来源的多个片段聚在一个围栏下，让 LLM 看清片段的来源归属。

### B) read_source —— 整篇阅读

按 `source_id` 或 facet 解析出**唯一一个**来源（0 或多匹配 → 返回 LLM 可见的 error 字符串，fail-closed，不做扇出——要读整季就让 LLM 并行发多个调用）。取它全部 `transcript_segments`，`format_turns`（含时间戳）拼成一段**连续的带说话人转录**，加一个 `Source [N]` + 摘要 + 时长 + 语言 的头。**没有逐块围栏**——连续性就是重点。转录为空 → 只回一句"无转录"的事实，连头都不给（避免拿标题编造）。

---
三、一句话总结

▎ **两个**检索工具：`find_passages`（确定性 facet 缩池 → 向量 ∥ BM25 → RRF 融合 → cross-encoder 精排 → gateless top-k）定位事实；`read_source`（单一来源 → 完整连续转录 + 内嵌时间戳）整篇阅读。LLM 在 stream 中自主选工具、自己决定停或继续，可观测指标通过 SSE 的 `rag` 事件送到前端（RetrievalLedger 组件）。

每一层都有降级路径（facet fail-open、FTS/向量互为后备、精排失败回退未精排），关掉哪个开关（rag.hybrid_enabled / reranking_enabled）都还能跑——默认开，但都可关。

▎ 中心赌注：在单用户、强模型、长文档的设定下，用**前沿模型的判断 + `read_source` 兜底 + grounding prompt** 取代相关度门 / relevance-critic。代价是硬例鲁棒性（ASR 噪声下的实体同一性、穷举式枚举）被接受为召回上限。
