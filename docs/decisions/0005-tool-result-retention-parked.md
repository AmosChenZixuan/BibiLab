# 0005 — Tool-result retention: built, measured, parked

**Status:** Step A (intra-turn dedup) Accepted. Step B (cross-turn retention) — implemented,
measured, and rejected on evidence, not merely deferred.
**Decided:** Step A: 2026-06-05, PR #435. Step B: investigated 2026-06-04 → ~2026-06-07,
implemented on branch `feat/find-passages-retention`, never merged.

## Step A — shipped

A turn-scoped `seen_chunk_ids` set threaded through `stream_with_tools` →
`execute_find_passages` prevents the same transcript chunk (keyed
`{source_id}_{int(ts_start)}_{int(ts_end)}`) from being rendered twice to the LLM within one
turn, across parallel or multi-hop `find_passages` calls. A fully-duplicate call collapses to
a one-line fact note. Pure token savings, no measured downside.

## Step B — the question, and why it was rejected

The chat pipeline's retention horizon is 0 turns: `expand_message_for_provider` drops every
`find_passages`/`read_section` tool exchange at every turn boundary, keeping only the
synthesized prose. This is already the most aggressive point on the tool-result-clearing
spectrum used by production agent frameworks generally (which typically clear old tool
results past a token threshold, not drop everything every turn) — the question investigated
was whether widening this to a bounded N-turn window would reduce redundant `find_passages`
calls without hurting quality.

A design and implementation were built (config flag, default 0; retained chunks rendered as
raw stored content) and a POC was run: N=1, 3 conversations × 2 runs, real LLM + real
retrieval. Findings:

- **Did not reduce tool-call count** — equal in most cases, *more* in some (retention traded
  `find_passages` calls for `read_source` escalation instead).
- **Enlarged per-turn context 60–130%.**
- **Measurably drifted the self-authored next-turn retrieval query** — retained terms leaked
  into the LLM's own query construction for the following turn.
- **No reduction in confabulation risk** — retention sometimes triggered *more* grounding
  calls, the opposite of the feared failure mode it was meant to guard against.

Verdict recorded at close: horizon-0 (the existing design) is the most cost-effective point on
the frontier evidence found; retention was not merged as default.

## Consequences held open

- The POC is explicitly self-flagged as directional only — tiny sample (N=1, 3 conversations),
  a free-tier model, no LLM-judge scoring. The "parked" verdict is provisional, not a closed
  question.
- Stated revisit triggers: conversation depth growing materially beyond current typical use,
  reliable prompt-prefix caching becoming available, or a round-trip-focused eval with an
  LLM judge — none of which had occurred as of this writing.
- The implementation branch (`feat/find-passages-retention`) is not merged; current
  `backend/src/bibilab/` carries no trace of the retention config or rendering code.
