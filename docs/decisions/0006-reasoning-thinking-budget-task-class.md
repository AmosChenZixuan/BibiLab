# 0006 — Reasoning-model thinking budget: reframe the task, not the parameter

**Status:** Accepted — scoped to the eval pipeline's fact-extraction step; not verified
against the production chat grounding prompt.
**Decided:** 2026-06-07, PR #439 (fixes issue #438).

## The problem

A POC against a reasoning model (`deepseek-v4-flash-free`, OpenAI-protocol proxy) showed the
eval pipeline's fact-extraction step burning its entire 32K token budget on the model
re-typing the source transcript into its own thinking tokens: `reasoning=32170`,
`finish=length`, `content=0`, 312 s wall time — an open-ended "extract up to 30 key facts"
prompt gave the model no natural stopping point.

Three parameter-level fixes were tried and rejected:

1. `reasoning_effort` hint — silently ignored by the proxy.
2. `response_format: json_object` — reduced but did not cap thinking volume.
3. `tool_choice: required` — the proxy rejected it outright when thinking mode was active.

A fourth option — forcing a different model — was rejected on principle: "we can't control
what model the user uses." LiteLLM was also considered and rejected: the endpoint is
OpenAI-protocol, so it would receive `reasoning_effort` (already shown to be ignored), never
Anthropic-style `budget_tokens` — a model-agnostic fix was required.

## The decision

Thinking volume is set by task class, not by any parameter:

> "extract/select/rank from a document has no natural stop → fills whatever budget you give
> it. Abstractive-summarize into N points is content-bounded (~4–12K observed) → reliably
> finishes under the ceiling."

`SOURCE_FACTS_PROMPT` was rewritten from open-ended extraction ("提取关键事实，最多30条") to a
bounded abstractive summary ("用 12-20 个要点概括"), keeping the same output schema. This is
not a control mechanism layered on top of the model — it changes the shape of what the model
is asked to do, so the existing token budget becomes sufficient by construction.

Verified on the live failing endpoint: 56 s wall time, 0 errors, non-empty content — versus
the original 312 s / 0 content.

## Scope — this was a narrower fix than it may sound

The "task-class lever" principle is stated generally, but the actual patch touched only the
eval pipeline's step-1 fact-extraction prompt (`eval/`), not the production chat path. Step-2
question generation and the grader were probed during the same investigation and found
already naturally-bounded (they have their own stopping conditions) — no change was needed
there.

## Consequences held open

Whether this principle has been applied — or even needs to be — in the production chat
grounding prompt (which also asks the model to synthesize an answer from retrieved fragments)
is not evidenced anywhere found. If a similar stuck-thinking failure surfaces in chat, this
decision is the template to reapply, not something already verified to cover it.
