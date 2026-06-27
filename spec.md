# #542 — Mindmap node evidence (provenance for chat)

## Goal
A click on a synthesized mindmap node (e.g. "近古纪元") should lead to a chat that can retrieve the underlying passages. Generation emits a per-node `evidence` (one short verbatim transcript quote); the click threads it into the chat query so BM25/vector retrieval has an in-corpus phrase to match.

## Scope
**In:** mindmap generation prompt + refine directives (backend); `evidence` threaded through the node→chat click chain (frontend) with a conditional reference-passage clause.
**Out (deliberately dropped from the original draft):** hover tooltip, 300-char truncation, null/string validation, 4-key i18n matrix, separate extraction pass, backfill. `MindMapResult.root` is an untyped `dict` so `evidence` rides through parse + render with zero backend model/parser/render change.

## Acceptance criteria
- **AC1** (backend, happy+empty): `evidence` on a node survives `_render_mind_map_markdown` into the artifact markdown verbatim; a node with no `evidence` renders identical to today.
- **AC2** (backend, regression guard): the generation prompt, `_MIND_MAP_SCHEMA_DIRECTIVE`, and `_MIND_MAP_INTEGRATE_DIRECTIVE` each instruct the LLM to emit / preserve `evidence` verbatim — so multi-batch refine doesn't paraphrase the quote away (the BM25 premise).
- **AC3** (frontend, happy+empty): the ask-in-chat message equals today's message when evidence is empty, and appends a reference-passage clause when evidence is present — for both root (no parent) and child (with parent) cases.
- **AC4** (frontend, wiring): clicking a node threads the node's `evidence` through MindMapBlock → ArtifactViewer → the ask-in-chat callback; nodes without evidence pass empty.

## Residual (UNVERIFIED here)
The ≥80% retrieval hit-rate eval on synthesized nodes needs a live LLM + local reranker/embed + golden lists, not runnable in this environment. Proxy verified: the full render/parse/click/compose path. Residual flagged for a human eval run.
