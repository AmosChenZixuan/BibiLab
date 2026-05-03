# RAG Retrieval Tuning (#220)

Methodology, results, and decisions from the retrieval calibration eval.

## Query Set

30 queries across 3 lists (38 videos total), 3 content types:

| List | Sources | Content type |
|------|---------|-------------|
| 黑塔 | 9 | Fantasy audio novel (narrative) |
| 干饭 | 13 | Cooking recipes (procedural) |
| AI面试 | 16 | AI/ML tech interviews (knowledge) |

| Category | Count | Example |
|----------|-------|---------|
| factual | 12 | "舒芙蕾需要几个鸡蛋？" |
| breadth | 9 | "哪些视频讲到了多轮对话优化？" |
| analytical | 6 | "比较Agent Skill和MCP的区别" |
| boundary | 3 | "这个list里一共有多少道菜？" (degenerate) |

27 queries used for retrieval scoring (#220); 3 degenerate queries (H8, G8, A8) probe classification limits (#234).

## Sweep Design

Two-axis sweep of tunable retrieval parameters:

| Axis | Values | What it controls |
|------|--------|-----------------|
| RRF k | 30, 60, 90 | Reciprocal Rank Fusion — blending weight between BM25 and vector rankings |
| rerank floor | null, -2.0, 0.0 | Minimum cross-encoder score for a chunk to survive into LLM context |

9 combos total (3 × 3). Each query runs all combos; union of chunks across combos is labeled once, then per-combo metrics are computed from the ranked lists.

**Retrieval pipeline:** hybrid search (BM25 + vector, RRF fusion, pool of 30) → cross-encoder rerank (bge-reranker-base) → floor filter → diverse top-k (per-source depth cap + total cap set by query type).

## Results

RRF_K values 30, 60, 90 produce bit-identical rankings at every floor (verified: 270 pairwise comparisons, 0 differences).
Only the three floor values produce distinct results:

### Overall (27 queries, 87 relevant)

| Floor | P@5 | P@10 | P@all | Recall | MRR |
|-------|-----|------|-------|--------|-----|
| null  | 0.200 | 0.165 | 0.148 | 0.480 | **0.559** |
| -2.0  | 0.283 | 0.282 | 0.284 | 0.432 | 0.472 |
| 0.0   | 0.322 | 0.321 | 0.321 | 0.375 | 0.466 |

### Per-list

**黑塔** (9 queries, 29 relevant):
| Floor | P@5 | P@10 | P@all | Recall | MRR |
|-------|-----|------|-------|--------|-----|
| null  | 0.244 | 0.228 | 0.228 | 0.487 | 0.611 |
| -2.0  | 0.283 | 0.297 | 0.297 | 0.515 | 0.611 |
| 0.0   | 0.289 | 0.287 | 0.287 | 0.465 | 0.611 |

**干饭** (9 queries, 25 relevant):
| Floor | P@5 | P@10 | P@all | Recall | MRR |
|-------|-----|------|-------|--------|-----|
| null  | 0.156 | 0.100 | 0.094 | 0.417 | **0.611** |
| -2.0  | 0.215 | 0.211 | 0.218 | 0.429 | 0.407 |
| 0.0   | 0.287 | 0.287 | 0.287 | 0.381 | 0.417 |

**AI面试** (9 queries, 33 relevant):
| Floor | P@5 | P@10 | P@all | Recall | MRR |
|-------|-----|------|-------|--------|-----|
| null  | 0.200 | 0.167 | 0.121 | 0.535 | **0.454** |
| -2.0  | 0.350 | 0.337 | 0.337 | 0.352 | 0.398 |
| 0.0   | 0.389 | 0.389 | 0.389 | 0.278 | 0.370 |

## Findings

### 1. RRF_K is a dead parameter (confirmed)

The cross-encoder reranks the entire candidate pool (30 chunks), completely overwriting the RRF fusion ordering. Verified during the calibration sweep by temporarily plumbing `rrf_k` through `retrieve()` → `hybrid_search()` → `_rrf_fuse()` and sweeping {30, 60, 90} × 3 floors. All 270 pairwise comparisons of ranked lists were bit-identical. RRF_K should be removed as a tunable (#245).

### 2. floor=null is the best default

A rerank floor of `null` (disabled) achieves the highest overall MRR (0.559). Floors trade recall for precision — the precision gain is real (P@5 from 0.200 → 0.322 with floor=0.0) but recall drop is larger (0.480 → 0.375), pulling MRR down.

Per-list behavior differs: 干饭 and AI面试 prefer null (MRR 0.611 and 0.454); 黑塔 has identical MRR across all floors — narrative content with highly distinct per-episode answers is robust to floor filtering. The practical guidance across lists is still null: it maximizes recall at minimal precision cost.

**BGE score distribution:** The bge-reranker-base model outputs logits in approximately [-10, +10]. A floor of 0.0 filters everything (negative scores dominate), and a floor of -2.0 filters ~15% of chunks.

The `rerank_min_score` default was set to `0.0` in code, which filtered all chunks for users with a stale config file. This was the root cause of the 0-chunk bug (8/30 queries returning empty). The code default is now `null` (commit bbe3d19).

### 3. Factual queries over-pad with diversity constraint

For focused (factual) queries, `params_for_type()` (shrunk in bbe3d19 to ensure source coverage) sets `top_k = max(base.top_k, min(sources_total, base.top_k * 3))`. For a 16-source list this means factual queries get `top_k=15` instead of the base `top_k=5`, pulling in noise from many sources when most factual queries only need 1-2 sources.

This is the current state for better or worse: it avoids the 0-chunk bug at the cost of precision. #250 will revisit whether the noise tradeoff is worth it — options include a tighter multiplier cap or per-type source-coverage weighting.

## Decisions

| Decision | Rationale | Issue |
|----------|-----------|-------|
| Keep `rerank_min_score=null` | Empirically best MRR; floor reduces recall | #220 |
| Remove RRF_K as tunable | Dead parameter — reranker overwrites RRF ordering | #245 |
| 3-bucket classifier sufficient | 93% accuracy on 30 queries; prompt fixes cover the gaps | #234 |
| Defer #228 classifier replacement | ~1 month, wait for real user query data | #228 |

## Caveats

- **27-query sample:** Small for statistical significance. Per-list breakdowns (9 queries each) are directional.
- **No LLM-end measurement:** Only retrieval metrics (P@K, Recall, MRR). End-to-end answer quality not measured — the reranker-cut chunks might have been "close enough" that the LLM could still answer.
- **Reranker-bound:** Floor values are specific to bge-reranker-base score distribution. A reranker swap (e.g. #225) invalidates floor calibration.
- **Label quality:** 356 chunks labeled manually by author with LLM pre-labeling assist. Single-labeler bias possible.

## Re-run Conditions

Re-run the eval sweep when:
- Reranker model changes (score distribution shifts)
- Major chunking changes (different candidate pool composition)
- Tool-calling search ships (#228) — verify no regression
- New lists added with different content characteristics

The eval harness was deliberately throwaway (author-specific corpus, single-use UI). To re-sweep: write a script that calls `retrieve()` directly, varying `cfg.rag.rerank_min_score` per call — that is the only parameter the campaign found to affect ranking. The candidate pool (30), reranker model, and RRF fusion are all fixed.
