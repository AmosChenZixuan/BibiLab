# 0004 — Reranker: model, quantization, and execution-provider pin

**Status:** Accepted — one intermediate proposal explicitly retracted mid-investigation for a
wrong premise; the CPU-latency limitation is accepted, not solved.
**Decided:** Chain of four stages, 2026-04-29 → 2026-06-28 (final PR #574, closing #567/#573).

## The decision as it stands

The reranker is `bge-reranker-base-q` (int8-quantized, bilingual XLM-RoBERTa cross-encoder,
`RERANKER_SPEC_ID`), 266 MB versus the fp32 model's 1061 MB. Its ONNX session sources
execution providers from `interpreting_providers()` (`backend/src/bibilab/pipeline/_shared.py`)
— an explicit CPU/CUDA-only allowlist that excludes compiler/JIT-based execution providers
(CoreML, etc.) rather than trusting ONNX Runtime's default provider-priority order.

## Evolution

| Stage | Date | Change | Why |
|---|---|---|---|
| 1 | 2026-04-29 (#202) | `sentence-transformers.CrossEncoder`, `ms-marco-MiniLM-L-6-v2` | Initial ship, English-only |
| 2 | 2026-04-30 (#210) | Hand-rolled ONNX runtime session, same English model | Dropped `sentence-transformers`/torch — dependency bloat |
| 3 | 2026-05-01 | Swapped to `bge-reranker-base` (XLM-RoBERTa), batched `predict()` | Bilingual zh+en support; one `session.run()` for all pairs instead of one per pair |
| 4 | 2026-06-28 (PR #574, closes #567/#573) | int8 quantization + `interpreting_providers()` pin | See incident below |

## The incident that drove stage 4

Issue #559 found the reranker session spiking to ~6.5 GB RSS on macOS, OOM-killing a 16 GB
worker. Root cause, narrowed in #567: `ort.get_available_providers()` puts
`CoreMLExecutionProvider` first on macOS by default; CoreML JIT-compiles the ONNX graph,
duplicating weights in the process. Worse, even the *quantized* model under CoreML measured
**>90 s per batch-30 rerank**, versus 1.15 s on CPU — CoreML fits memory trivially (662 MB)
but is catastrophically slow on this graph shape.

An intermediate proposal (#573) suggested hardcoding `providers=["CPUExecutionProvider"]` and
deleting `interpreting_providers()` as dead code. This was **explicitly retracted**
mid-investigation:

> "Retracted — wrong premise. This proposed a host-derived fp32-on-GPU design from a
> benchmark that installed onnxruntime-gpu... GPU-onnx is not a direction."

PR #574 shipped a smaller, different fix — reusing `interpreting_providers()` for the
reranker (previously it had one caller) instead of deleting it:

> "This differs from issue 573 as written... Issue 573's 'delete it (zero callers)' was
> circular... Reused it for the reranker instead → two callers, one-line fix, nothing deleted."

## Measured tradeoffs (i9-14900F, CPU)

| Batch (effective_top_k) | fp32 | int8 |
|---|---:|---:|
| 10 | 1.58 s | 0.95 s |
| 30 | 4.79 s | 2.64 s |
| 60 | 10.1 s | 5.27 s |

Quantization quality check against fp32 ordering: top-3 identical, top-8 91% overlap,
Kendall's τ +0.88 — accepted as a safe ordering-quality tradeoff, consistent with
[0002](0002-retrieval-gate-deleted.md)'s "rerank is ordering, not authority" — a reshuffled
tail costs less here than it would under a gate that trusted rerank order as ground truth.

## Consequences held open

**Not solved, only made survivable.** At batch ≥30, CPU latency (2.64–5.27 s int8) still does
not fit a 1–2 s interactive budget. No further fix has shipped for this; it is accepted
because the retrieval architecture ([0002](0002-retrieval-gate-deleted.md)) does not depend
on rerank being fast, only on it being roughly ordering-correct.
