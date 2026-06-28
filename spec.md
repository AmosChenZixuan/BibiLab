# #567 — Quantized reranker (primary macOS OOM fix) + registry sizes + CoreML gate

## Goal

Register the int8-quantized `bge-reranker-base` ONNX model, make it the config-selected
runtime default reranker (fp32 opt-in), correct wrong `size_mb` metadata, dedup the
duplicated path logic in `rerank.py`, and measure whether the quantized model fits 16 GB
under CoreML on macOS — the gate that decides whether #559 (reranker CPU pin) ships.

## Design (resolved via grill-me)

1. **Gate** — agent runs the CoreML peak-RSS + batch 30/60 latency probe on this Mac.
2. **Default scope** — global quantized default, all platforms (int8 is also faster on CPU).
3. **Selection** — new `RagConfig.reranker_spec_id` field, default quantized; `ensure()`
   downloads only the selected spec (one resident). fp32 opt-in by setting the field.
4. **Acceptance for ordering** — manual smoke check (fp32 vs int8 top-k on one query). The
   hardware-determinism invariant survives (int8 identical across boxes); absolute-vs-fp32
   quality is a separate, accepted tradeoff.

`RERANKER_SPEC_ID` module constant becomes dead once selection routes through config →
removed (code-health rule #1). `RagConfig.reranker_spec_id` is the single runtime source.

## Scope

**In**
- `model_registry.py`: register `bge-reranker-base-q` spec (http `onnx/model_quantized.onnx`
  stored as normalized `model.onnx` + `tokenizer.json`, distinct `local_subdir`,
  `size_mb=266`); fix `bge-reranker-base` `280→1061`, `multilingual-e5` `420→449`; remove
  `RERANKER_SPEC_ID` constant + `__all__` entry; `required_models` reads `cfg.rag.reranker_spec_id`.
- `config.py`: add `RagConfig.reranker_spec_id: str = "bge-reranker-base-q"`.
- `rerank.py`: `ONNXCrossEncoder` consumes the dir **returned by** `ensure(spec_id)`; delete
  `_model_dir()` + `_MODEL_REPO`; read spec id from config; rewrite the now-false
  "intentionally fixed / configurable would break the ordering invariant" comment.
- `routers/health.py`: `_check_reranker_model` reads `cfg.rag.reranker_spec_id`.
- Docs: `backend/CLAUDE.md` ("Reranker model is fixed" + config-schema JSON), root `CLAUDE.md`
  Retrieval row reranker mention.

**Out**
- High-batch rerank latency itself (multi-second on CPU regardless) — #369/#530.
- README RAM line — pending post-fix macOS peak (#559).
- `multilingual-e5` id rename (misnomer) — risks persisted config keys; left.

## Acceptance Criteria

**AC1 — registry sizes corrected.** (happy)
`get_spec("bge-reranker-base").size_mb == 1061` and `get_spec("multilingual-e5").size_mb == 449`.
Observable: unit test on spec metadata.

**AC2 — quantized spec registered.** (happy)
`get_spec("bge-reranker-base-q")` returns a reranker spec: `size_mb == 266`, `http_files`
URL ends `onnx/model_quantized.onnx` mapped to rel-path `model.onnx`, `integrity_files`
include `model.onnx` + `tokenizer.json`, `local_subdir` distinct from fp32's.
Observable: unit test on spec metadata.

**AC3 — config-selected default, drift-guarded.** (happy + error)
happy: `RagConfig().reranker_spec_id == "bge-reranker-base-q"` and
`get_spec(RagConfig().reranker_spec_id)` resolves (default names a real registered spec).
error: `RERANKER_SPEC_ID` no longer importable from `model_registry` (constant removed).
Observable: unit tests.

**AC4 — rerank loads the dir `ensure()` returns (dedup).** (happy)
`ONNXCrossEncoder` builds its session from the path `ensure(<configured spec>)` returns, not
a recomputed dir; `_model_dir`/`_MODEL_REPO` are gone. A config pointing at the quantized spec
loads from the quantized `local_subdir`. Observable: unit test patching `ensure`/`ort` and
asserting the session path equals the `ensure` return; `grep -c "_model_dir\|_MODEL_REPO"
rerank.py == 0`.

**AC5 — quantized downloads, builds, sane ordering.** (happy; network-bound)
The quantized spec downloads and builds a real ONNX session; on a known query its top-k is
sensible and not nonsense-reordered vs fp32. Strongest in-env proxy: real download + real
session build + real `predict()` on a fixed query/doc set, asserting the relevant doc ranks
top. UNVERIFIED residual if network/model unavailable here — flagged in reflection.

**AC6 — CoreML gate measured + reported.** (happy; macOS-bound)
On this Mac: quantized reranker CoreML session-build peak RSS (`ru_maxrss`) + batch 30/60
rerank latency under CoreML vs CPU, measured single-foreground-process. Posted as a #567
comment with the fits-16GB verdict and #559 recommendation. Strongest in-env proxy: run the
real probe here. UNVERIFIED residual for anything the box can't reproduce (e.g. concurrent
embed 3.4 GB resident) — stated plainly; #559 close/unblock left to the human.

**AC7 — suite green.** `uv run pytest` passes.
