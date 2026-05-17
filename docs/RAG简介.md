---
一、入库阶段（一个视频从音频到"可被搜"的过程）

文件入口：backend/src/bibilab/pipeline/

1. 转录 (transcribe.py)
Whisper 把音频转成一堆带时间戳的小 segment（5–15 秒一段）。
2. 切片 (chunk.py)
贪心合并相邻 segment，凑到 ~300 token 就封一个 RagChunk，超过 400 token 的单段直接独立成块。每块记下 timestamp_start/end 和 sequence_index。
目的：让一块里的内容是一个完整的语义片段，而不是 5 秒一段太碎。
3. 双轨入库 (embed.py: embed_chunks)
每个 chunk 同时写两份索引——这是这次最大的结构性改动：

    - 向量库（ChromaDB）：用本地 ONNX MiniLM 模型把文本变成向量，存进 ~/.bibilab/chroma/，metadata 带 video_id / list_id / video_title / 时间戳。
    - 关键词库（SQLite FTS5）：populate_fts() 把同一批 chunk 文本插入 chunks_fts 虚拟表，让 SQLite 的 BM25 能直接打分。

两份都按 video_id 幂等，重新跑会先 delete 老数据再插。

▎ 为什么写两份：向量擅长语义近似（"他在讲什么意思"），BM25 擅长精确字面（"他有没有说过 XX 这个词"）。两边各有失误的场景，所以后面查询要混搭。

---
二、查询阶段（用户在聊天里问一句的全过程）

文件入口：routers/chat.py → pipeline/embed.py: retrieve() → pipeline/rerank.py

1. 混合检索（核心）—retrieve() → hybrid_search()

              ┌──> query_chunks (向量, top 30)
  query ──┤                                    ├──> RRF 融合 ──> 30 个候选
              └──> query_fts    (BM25,  top 30)

- 并行两条腿一起跑（asyncio.gather）。
- RRF 融合（_rrf_fuse，k=60）：每个 chunk 在两个榜里的排名取 1/(k+rank)，加起来得分越高越靠前。这样无需把"距离"和"BM25 分"强行换算到同一尺度。
- 降级：FTS 出错或空 → 用纯向量；向量出错 → 用纯 FTS。

2. 精排（cross-encoder 重排）—rerank.py

把 30 个候选和原 query 拼成 pair，丢给 cross-encoder/Xenova/bge-reranker-base-v2（懒加载、单例）打一个真正"这句和那句相不相关"的分数，取 top_k（默认 5）。
重排失败也不会崩，回退到未重排的前 5。

▎ 双塔向量是"两句话各自变向量再算余弦"，跨编码器是"两句话拼一起一起编码"，更准但更慢——所以只用来精修头部 30 个，不全量跑。

3. 检索参数来自 LLM 调用

LLM 在 stream 中判断问题类型后调用 `retrieve` 工具，参数通过 `search_mode_to_params` 映射为 RetrievalParams（factual/breadth/analytical 三键）。

Focused 模式：直接给 LLM 那 5 个 top chunk（深度 > 覆盖）。
Broad 模式：用 best_by_source 字典，每个视频只保留它内部最相关的那一块，按分数排序输出（覆盖 > 深度）。

4. 检索范围语义（#295）

`retrieve` 工具支持 `source_ids`（白名单）和 `exclude_source_ids`（排除黑名单）。两者并存时 blacklist 优先。白名单为空时默认搜全部。

---

三、一句话总结这次更新做了什么

▎ 老版本：单路向量召回 → 直接喂给 LLM。
▎ 新版本：(向量 ∥ BM25) → RRF 融合 → cross-encoder 精排 → LLM 通过工具调用决定检索范围和模式，并把可观测指标通过 SSE 的 `rag` 事件送到前端（RetrievalLedger 组件）。

每一层都有降级路径，关掉哪个开关（rag.hybrid_enabled / reranking_enabled）都还能跑——所以新功能默认开但都可关。
