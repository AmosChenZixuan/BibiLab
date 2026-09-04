# 0003 — Bounded sections: a token-bounded tier between sources and chunks

**Status:** Accepted.
**Decided:** 2026-06-07 (design), shipped via EPIC #457 (sub-issues #452/#453/#454/#455/#456,
PRs #458/#460/#461/#462+#463/#459), source-level digest columns dropped by a follow-up
cleanup PR (#465).

## The decision

A source stays exactly one row — `video_id` dedup, `[N]` citation unit, facets, and list UI
all assume one video = one source, and this never changes. Internally, a source becomes an
ordered set of `Section` rows: contiguous, token-bounded spans (target 12000 tokens, zone
[7200, 16800]) snapped to the longest in-zone pause, each carrying its own summary and
keywords, generated via a **refine chain** (each section's summary is generated with the
running context of prior sections, not independently in parallel) because adjacent sections
are usually topically continuous. Short videos (the common case) collapse to exactly one
section — no behavior change on the common path.

`sources.summary`/`sources.keywords` were deleted outright once every reader (digest worker,
section UI, chat) migrated onto `sections` — no source-level digest cache was kept.

## The problem this solved

Three failure modes of unbounded transcript length:

1. **Digest quality** — one ~150-word summary over an entire long transcript is
   information-lossy; keyword extraction over a long transcript risks output-budget overflow.
   Measured on a 97-minute synthetic transcript (6 real transcripts concatenated): a single
   whole-transcript digest produced 8 generic keywords versus 76 specific, queryable keywords
   from 14 per-section summaries on the same content.
2. **Chat context exhaustion** — `read_source` (the predecessor to `read_section`) pulled the
   *entire* transcript into context, unbounded.
3. **Artifact generation** — the artifact pipeline read all selected sources in full,
   unbounded both per-source and in source count.

This tier is also the mechanism that resolves issue #396 (coverage-question confabulation,
see [0002](0002-retrieval-gate-deleted.md)'s sibling decision at the tool-surface layer): with
per-section outlines available to `find_passages`, the LLM has a bounded, complete map of the
source instead of needing to guess whether it has "seen everything."

## Boundary algorithm: chosen by POC, not assumed

A comparison on the same 97-minute synthetic transcript (episode seams from the source
material = ground truth) tested {time-target(~30min), token-target(~6000)} ×
{pause-signal, embedding topic-drift}:

| Strategy | Seam recovery | Cost |
|---|---|---|
| Time-target | Didn't bound tokens (one 13K-token section produced) | — |
| Embedding topic-drift | 0/5 | Full extra embedding pass |
| Token-target + longest-pause | **4/5** | Zero extra cost (reuses existing pause signal) |

Token-target + pause won outright — cheaper and more accurate. The pause signal reused here
is the same one chunking already computes (pause-aware sentence-boundary detection, frozen
since 2026-05-30) — this design deliberately declined to build a second, more expensive
signal (e.g. embedding-based topic segmentation, the RAPTOR-style approach) when a cheaper
existing one already worked.

## Target-token tuning

The target was tuned from an initial 6000 to the shipped 12000: at 6000, the max
sub-20-minute zh video in the corpus (~6200 tokens) exceeded the backstop and got needlessly
split into 2 sections. 12000 keeps the common case (short videos) at exactly 1 section.

## Consequences held open

- `sections` boundaries are computed once at digest time; re-deriving them without also
  re-chunking (or vice versa) can produce boundaries that straddle existing chunks — later
  design work (for #453) explicitly guards against this by keeping rerun digest-only.
- The refine chain trades cost (serial, not parallelizable across sections) for coherence;
  accepted because long sources needing more than 1-2 sections are rare in the current corpus.
