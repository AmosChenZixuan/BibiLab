# 0002 — Retrieval gate deleted: rerank orders, does not authorize

**Status:** Accepted — challenged once (POC #530), reconfirmed with stronger evidence.
**Decided:** 2026-06-02 (EPIC #369, PR #394). **Reconfirmed:** 2026-06-13 (issue #530, closed
NO-GO on reopening).

## The decision as it stands

`find_passages` (`backend/src/bibilab/pipeline/embed.py`) returns the top-8 hybrid-search
results by cross-encoder rerank order, with no relevance/quantile gate, no per-source
diversity cap, and no neighbor-pull. Docstring, `embed.py:534-539`:

```python
"""Recall-biased locator: hybrid search → rerank → top_k by rerank order.

No relevance gate (rerank is ordering, not authority) and no per-source
diversity cap. The LLM filters relevance and decides whether to escalate
to read_section.
"""
```

The LLM, not the backend, judges whether the returned fragments are sufficient or whether to
escalate to a verbatim `read_section` call.

## What was removed, and why

A relevance gate (`_quantile_gate`) existed before this decision: `threshold =
max(median(scores), top − RELEVANCE_MARGIN)`, `RELEVANCE_MARGIN = 4.0`, added 2026-05-12
(#277) with its margin explicitly unvalidated ("relies on telemetry post-ship, not offline
sweeps"). It was deleted 2026-06-02 as part of EPIC #369's redesign, along with
`_diverse_top_k` and `_adaptive_depth`.

Two production failures justified removal (issue #369, closing comment 2026-06-01):

1. **Homophone entity** (`前美子是谁` → should resolve to 钱美子): the locator recalled the
   right source into the candidate pool, but the reranker is homophone-blind (it scores raw
   hanzi), and the gate collapsed 24 candidates to 1 — dropping the correct source and
   keeping a misleading one. The forced `read_source` escalation then read the wrong source.
2. **Abstraction/term mismatch** (`隐身术的施法三要素`): the right source surfaced, but rerank
   score was *anti-correlated* with correctness — a junk chunk scored 5.37 against the gold
   chunk's 0.81.

A replay of both cases under {gate on/off} × {diversity on/off} reproduced the production
failure exactly with the gate on, and recovered gold at rank 2 of top-8 with it off.

## Central bet

Standard hybrid+RRF+rerank+top-k, no bespoke gate; the stop/continue judgment ("are these
fragments enough, or should I escalate?") is assigned to the LLM via the grounding prompt,
not to a structural mechanism. Rejected alternatives at decision time: an entity-coverage
detector, a hit-distribution router (both judged "the LLM's job"), and a CRAG/Self-RAG-style
relevance critic (deferred, never built).

## The 2026-06-13 re-litigation (issue #530)

The gate's absence was reopened as a live question, driven by measured "waste" — 46% of
surfaced sections across 83 sections / 14 turns / 3 corpora were never cited or read. The
go/no-go bar: recall ≥95% overall and 100% on enumeration-regime turns, AND ≥50% of the
unused tail removed, from any cut of the form `keep iff score ≥ turn_top − δ` (the only
legitimate form, since the reranker's scores are uncalibrated logits).

**Result: NO-GO.** No `δ` satisfies both constraints — recall≥95% caps waste-removed at 38%;
waste≥50% caps recall at 67%. The decisive finding was a **structural score inversion**, not
merely an insufficient margin:

> "6 of 14 turns have `max(unused score) > min(used score)` — the reranker ranks an unused
> section above a used one... When the ranking itself inverts usefulness, any monotonic cut
> that drops the unused section also drops a used one."

Example inversions (turn, used-min vs unused-max): `归树是谁` 4.49 vs 5.12; `念刃是什么` 1.50
vs 2.61. Alternatives considered and rejected in the same close: reranker calibration
(Platt/isotonic — "monotonic, so it cannot fix inversions"), sharper queries (the inversions
occur on already-clean entity queries, so query hygiene doesn't dissolve them), and a
CRAG-style relevance critic ("it only relocates the cost of reading the score, rather than
reading it off the existing one").

Caveat on record from the same close: n=14 turns / 83 sections is "directional, not
definitive," but the failure mode is structural (rank inversions), not a marginal aggregate
gap — and since `used = cited ∪ read` is a lower bound on usefulness, the measured waste is
an upper bound, not an exact figure.

## Consequences held open

- The removed gate's own margin (`RELEVANCE_MARGIN = 4.0`) was itself never validated before
  removal — this decision replaces a known-unvalidated baseline, not a proven one.
- Gateless top-8 depends on the reranker being homophone-blind and occasionally
  anti-correlated staying tolerable — see [0004](0004-reranker-model-quantization-ep-pin.md)
  for the reranker's own evolution and accepted latency/quality tradeoffs.
- One corpus (a novel the underlying LLM has memorized parametrically) was excluded from the
  #530 gate as contaminated, reported for reference only — not part of the pass/fail evidence.
