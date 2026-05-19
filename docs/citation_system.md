# Citation System (V1)

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
| `backend/src/bibilab/pipeline/chat_tools.py` | `CitationRegistryEntry`, `_format_chunk_for_llm`, `_build_source_headers`, `_build_fenced_chunks`, `execute_retrieve` |
| `backend/src/bibilab/pipeline/citation_parser.py` | `parse_delta`, `flush_buffer` — incremental regex parser |
| `backend/src/bibilab/routers/chat.py` | `SSE_EVENT_CITATION`, `stream_with_tools` registry+parser integration, `chat_endpoint` content_blocks persistence |
| `backend/src/bibilab/pipeline/chat_summary.py` | Compression prompt (citation preservation removed) |
| `web/src/lib/chat-utils.ts` | `ContentBlock` type, `RagSource` (with `source_id`), `stripLegacyTokens` |
| `web/src/components/lists/hooks/useSSEStream.ts` | SSE consumer — accumulates `ContentBlock[]` |
| `web/src/components/lists/hooks/useConversationHistory.ts` | `MessageUI` with `contentBlocks`, legacy fallback |
| `web/src/components/lists/ChatPanel.tsx` | `CitationChip` component, block-based renderer |
| `web/src/pages/ListDetailPage.tsx` | `handleOpenSource` with `highlightChunks` opts (V1: accepted, unused) |

## Data flow

1. User sends message → `chat_endpoint` builds `source_map` (video_id → source_id)
2. `stream_with_tools` starts with empty `registry: dict[str, CitationRegistryEntry]`
3. LLM calls `retrieve` → `execute_retrieve` assigns indices, formats chunks with `[N @ Ts-Ts]`
4. Tool result includes `Source [N]: "Title"` headers + enumeration line; chunks grouped under per-source `===== Source [N]: "title" =====` fences (#297)
5. LLM responds with deltas containing `[N]` tokens
6. `CitationParser` strips `[N]`, emits `citation` SSE events with `{index, source_id}`
7. Frontend assembles `ContentBlock[]` from interleaved delta + citation events
8. On stream end: `content_blocks` persisted in `metadata`, `source_coverage` ordered by registry index

## Chunk ID format

`chunk_ids` are synthetic strings `"{video_id}_{int(start_seconds)}_{int(end_seconds)}"`,
not foreign keys to `chunks` table. Built in `execute_retrieve`. V2 highlighting
must re-derive timestamps by parsing these strings; consider replacing with
real chunk UUIDs if and when chunks gain stable IDs in the embedding pipeline.

## V2 follow-up

- LLM prompt extension for chunk-specific citations (`[N#M]`)
- Hover snippet preview: lazy-load chunk text
