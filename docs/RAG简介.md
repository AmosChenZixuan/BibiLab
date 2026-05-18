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

LLM 在 stream 中自己决定要不要查、查什么——它直接调用 `retrieve` 工具（不再有独立的分类器前置调用）。一次 `retrieve` 经过四层：

1. 确定性 facet 缩池（检索前的唯一收窄手段）

`retrieve` 接 `sequence_number` / `season_number`（整数，可选）。LLM 只负责从问题里**抽取**这两个数（解析，可靠），后端拿它们和 `sources` 表里的列做**精确匹配**（集合运算，确定），在混合检索**之前**把全量来源池缩到匹配子集。

▎ Fail-open 是硬规则：没传 facet → 不筛；传了但零匹配（或 facet 子查询 DB 报错）→ **退回全量来源池，绝不清空**，打 `facet_scope.no_match=true`，并在 LLM 可见的工具结果里加一行提示，让模型先声明"按全部来源搜索"再作答。`scoped_pool_size` 记全量池大小，narrowed 数量看 `facet_scope.matched_count`。#310 顺带删掉了 `source_ids`/`exclude_source_ids` 索引决策和系统提示里的来源清单——系统提示因此变成 per-language 静态文本，Anthropic prompt cache 能覆盖整个前缀。`series_name` 的模糊匹配推迟到单独 follow-up。

2. 混合检索 — retrieve() → hybrid_search()

              ┌──> query_chunks (向量)
  query ──┤                              ├──> RRF 融合 ──> 候选池
              └──> query_fts    (BM25)

- 并行两条腿一起跑（asyncio.gather）。
- 候选池大小是动态的：`_dynamic_pool(n) = min(max(n×3, 10), 60)`，再用 `max(pool, top_k)` 兜底——按来源数 3× 伸缩，小列表保底 10，大列表封顶 60（控延迟）。
- RRF 融合（_rrf_fuse，k=60）：每个 chunk 在两个榜里取 1/(k+rank) 累加；无需把"向量距离"和"BM25 分"换算到同一尺度。
- 降级：FTS 出错或空 → 纯向量；向量出错 → 纯 FTS。

3. 精排（cross-encoder 重排）— rerank.py

把候选和原 query 拼成 pair，丢给 `Xenova/bge-reranker-base`（XLM-RoBERTa，中英双语；懒加载、单例）打一个真正"这句和那句相不相关"的分数。重排失败也不会崩，回退到未重排的候选。

▎ 双塔向量是"两句各自变向量再算余弦"，跨编码器是"两句拼一起一起编码"，更准但更慢——所以只精修候选池头部，不全量跑。

4. 相关度门 + 多样性截断 — _quantile_gate / _diverse_top_k

- 分位门：`threshold = max(median(分数), top − margin)`，只在重排成功时跑。`margin` 不是常数，由 LLM 传的 `expected_hits` 桶决定（见 rag_tuning.md）：

  | expected_hits | margin | 检索参数 | 意图 |
  |---|---|---|---|
  | one | 1.0 | depth=1, top_k=2 | 单一事实，只留最相关 |
  | few | 2.0 | depth=2, top_k=8 | 默认，适度铺开 |
  | many | 2.5 | depth=5, top_k=24 | 综述/总结，放宽 |

- 多样性截断 `_diverse_top_k`：每个来源的入选块数有上限（depth_per_source），总数有上限（top_k），避免单一视频刷屏。
- `rerank_min_score` 静态地板已废弃（#277，no-op + 启动告警），分位门取代它。`RetrievalResult.gate_margin` 记录实际用的 margin，供生产流量按桶审计。

---
三、一句话总结

▎ 老版本：单路向量召回 → 直接喂给 LLM。
▎ 新版本：LLM 中途自调 `retrieve` →（确定性 facet 缩池）→（向量 ∥ BM25）→ RRF 融合 → cross-encoder 精排 → 分位门 + 多样性截断 → 把可观测指标通过 SSE 的 `rag` 事件送到前端（RetrievalLedger 组件）。

每一层都有降级路径，关掉哪个开关（rag.hybrid_enabled / reranking_enabled）都还能跑——新功能默认开但都可关。
