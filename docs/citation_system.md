# Citation System

## Overview

NotebookLM-style inline citations. The LLM cites sources using `[N]` tokens
embedded in prose. The backend parses these tokens server-side, emits structured
`citation` SSE events interleaved with stripped text deltas, and persists the
interleaved structure as `metadata.content_blocks`. The frontend renders inline
numbered chips with hover tooltips and click-to-open-source.

## Architecture

```
LLM delta stream
  → CitationParser (regex [\d+], validates against registry)
  → SSE: interleaved "delta" + "citation" events
  → Frontend: assembles ContentBlock[]
  → Render: ReactMarkdown for text, CitationChip for citations
  → Persist: metadata.content_blocks for history reload
```

## Key files

| File | Role |
|---|---|
| `backend/src/bibilab/pipeline/chat_tools.py` | `CitationRegistryEntry`, `_format_chunk_for_llm`, `_build_source_headers`, `_build_fenced_chunks`, `execute_find_passages`, `execute_read_source` |
| `backend/src/bibilab/pipeline/citation_parser.py` | `parse_delta`, `flush_buffer` — incremental regex parser |
| `backend/src/bibilab/routers/chat.py` | `SSE_EVENT_CITATION`, `stream_with_tools` registry+parser integration, `chat_endpoint` content_blocks persistence |
| `backend/src/bibilab/pipeline/chat_summary.py` | Compression prompt (citation preservation removed) |
| `web/src/lib/chat-utils.ts` | `ContentBlock` type, `RagSource` (with `source_id` + `title` only), `stripLegacyTokens` |
| `web/src/components/lists/hooks/useSSEStream.ts` | SSE consumer — accumulates `ContentBlock[]`, dispatches `find_passages` + `read_source` `tool_result` events |
| `web/src/components/lists/hooks/useConversationHistory.ts` | `MessageUI` with `contentBlocks` + `rag`, legacy fallback |
| `web/src/components/lists/ChatPanel.tsx` | `CitationChip` component, block-based renderer |
| `web/src/pages/ListDetailPage.tsx` | `handleOpenSource` with `highlightChunks` opts (accepted but unused) |

## Data flow

1. User sends message → `chat_endpoint` builds `source_map` (video_id → source_id)
2. `stream_with_tools` starts with empty `registry: dict[str, CitationRegistryEntry]`
3. LLM calls `find_passages` (locator) or `read_source` (whole-source) → `execute_find_passages` / `execute_read_source` assign indices, format body
4. Tool result includes `Source [N]: "Title"` headers + enumeration line; chunks grouped under per-source `===== Source [N]: "title" =====` fences
5. LLM responds with deltas containing `[N]` tokens
6. `CitationParser` strips `[N]`, emits `citation` SSE events with `{index, source_id, chunk_ids}` (`chunk_ids` empty for `read_source`)
7. Frontend assembles `ContentBlock[]` from interleaved delta + citation events
8. On stream end: `content_blocks` persisted in `metadata`, `rag.calls` ordered by registry index

## Chunk ID format

`chunk_ids` are synthetic strings `"{source_id}_{int(start_seconds)}_{int(end_seconds)}"`,
not foreign keys to `chunks` table. Built in `execute_find_passages`. Highlighting
must re-derive timestamps by parsing these strings; consider replacing with
real chunk UUIDs if and when chunks gain stable IDs in the embedding pipeline.
Only `find_passages` emits `chunk_ids`; `read_source` citations carry none (see
below).

## read_source citation grain (coarse, source-level)

`read_source` returns one source's full continuous transcript, not ranked
chunks — so its citation is **coarse**: a single `[N]` stands for the whole
source.

- **One `[N]` per source.** `execute_read_source` allocates (or reuses) a
  `CitationRegistryEntry` keyed by `source_id`; it populates `index`, `source_id`,
  `title` only — **`chunk_ids` stays empty**. The narrative body is a continuous
  speaker-turn transcript (`format_turns`, inline `[MM:SS]`), not a fenced chunk
  list, so there is no per-chunk anchor to cite.
- **In prose the LLM cites moments by timestamp + source:** `"在 12:30 X 发生了
  [1]"` — the `[1]` resolves to the source; the `12:30` (visible in the body) is
  how the user locates the moment. Per-chunk click-to-seek is **not supported**
  (deferred; would need an `[N.M]` chunk-level citation scheme).
- **Shared `[N]` counter with `find_passages`, dedup by `source_id`.** If
  `find_passages` already registered a source this turn, `read_source` reuses the
  same `[N]` (dedup by `source_id`); otherwise it allocates `max(index)+1`. So a source read
  in full after being located keeps one stable number.
- **Frontend renders a source-level chip** (`RagSource` carries `source_id` +
  `title` only); the empty `chunk_ids` means no highlight payload, which the
  renderer treats as best-effort (`web/CLAUDE.md`).
- **Cross-turn reseed is find_passages-only.** `reseed_citation_registry` walks
  stored `tool_blocks` whose name is in `RETRIEVE_TOOL_NAMES` (= `find_passages`)
  and rebinds their chunk-bearing `[N]`. A prior-turn `read_source` `[N]` (no
  chunk array) is not reseeded; if the same source is re-registered this turn it
  gets a fresh number, otherwise the legacy marker in old prose is inert text —
  acceptable, since live citations only need to survive on post-window messages.

## Deferred

- LLM prompt extension for chunk-specific citations (`[N#M]`)
- Hover snippet preview: lazy-load chunk text
