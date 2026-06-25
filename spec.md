# #557 — outline-only citations render as empty "" + dead [N] in chat ledger

## Goal

Outline-only `find_passages` results render cleanly in the chat ledger: no
empty `""` for the preview, no `0.00` for the score, and `[N]` citations in
the answer text are clickable and jump to the section opening. The chunk-hit
rendering path is unchanged.

## Scope

### In scope

- `backend/src/bibilab/pipeline/chat_tools.py`
  - Outline expansion loop: set `entry.citable = True` and
    `entry.preview = summary_by_section_id[section_id]` for outline-only
    sections (those without chunk hits).
  - Update `CitationRegistryEntry.citable` field docstring to reflect the
    new semantic.
  - Update the outline-loop comment to reflect the new semantic.
- `backend/src/bibilab/routers/chat.py`
  - Context reconstruction: drop `or 0.0`/`or ""` coercions for
    `rerank_score` / `timestamp_start` / `timestamp_end` / `preview` /
    `chunk_id`. Pass values through as `None` for outline-only fields.
- `web/src/lib/chat-utils.ts`
  - Widen `RetrievalChunk.rerank_score` to `number | null`.
  - Widen `RetrievalChunk.chunk_id` to `string | null`.
  - `RetrievalChunk.preview` stays `string` (always populated now).
- `web/src/components/lists/ToolLedgerRow.tsx`
  - Use `chunk.section_id` (always populated) as the React key.
  - Render `chunk.rerank_score` only if non-null (no trailing `·`).
  - Render `chunk.preview` as-is (no skip logic).

### Out of scope

- Ledger header count label "引用 N 个片段" miscounts outline sections → #556.
- LLM prompt guidance — the LLM is already correctly citing outline
  summaries for overview questions (the issue's "dead [N]" evidence); no
  prompt update is needed.
- New tests for the outline rendering behavior — deferred to a follow-up
  (skipped per the grill-me session). AC1 verification is by code review
  and manual smoke; AC2 is covered by the existing chat flow tests.

## Files touched

- `backend/src/bibilab/pipeline/chat_tools.py` (3 hunks)
- `backend/src/bibilab/routers/chat.py` (1 hunk)
- `web/src/lib/chat-utils.ts` (1 hunk)
- `web/src/components/lists/ToolLedgerRow.tsx` (1 hunk)

## Acceptance criteria

### AC1 — Outline-only rendering (happy)

**Branch:** happy (outline-only result).

**Observable:** When `find_passages` returns outline-only sections (facet
match, no chunks), the `rag.calls[].context[]` payload contains entries
where:

- `rerank_score: null`
- `preview: <section summary text>` (non-empty)
- `chunk_id: null`
- `timestamp_start` / `timestamp_end`: section's `[start, end]` (not 0:00)
- `section_id` / `section_seq` / `source_id` / `source_title` /
  `citation_index`: populated

The frontend ledger renders the entry as
`[N] title · 0:00–27:01 · §1` followed by the section summary text, with no
trailing `·` and no `0.00`. The `[N]` in the answer text is clickable and
jumps to the section opening in the source viewer.

**Verification:** code review + manual smoke of the rendered ledger
(no new CI test — deferred to follow-up per the grill-me session).

### AC2 — Chunk-hit rendering (unchanged)

**Branch:** happy (chunk-hit result).

**Observable:** When `find_passages` returns sections with chunk hits, the
`rag.calls[].context[]` payload is unchanged from the pre-fix behavior:

- `rerank_score: <chunk's rerank score>` (number)
- `preview: <chunk's verbatim body>` (non-empty)
- `chunk_id: <chunk id>` (non-null)
- `timestamp_start` / `timestamp_end`: chunk's `[start, end]`

The frontend ledger renders the entry as
`[N] title · 0:00–27:01 · §1 · 0.87` followed by the verbatim quote text,
with the trailing `·` and the score.

**Verification:** existing chat flow tests cover the chunk-hit rendering
path (the fix is additive for outline-only entries and does not change
chunk-hit behavior).

### AC3 — Existing test suite + lint green

**Branch:** happy (no regressions).

**Observable:** `uv run pytest` and `npm test && npm run lint` exit 0 with
no new failures. No existing behavior is changed for the chunk-hit path.

**Verification:** run the full test suite.
