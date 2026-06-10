# Citation System

> **Section-grained since PR #462** (merged 2026-06-09). The `[N]` citation unit
> is a **section** of a source, not a source. The LLM's `read_section(section_id="[N]")`
> returns one section's bounded transcript. The facet→full-episode outline path
> emits every section of a matched source as its own `[N]` (orientation summaries;
> non-citable until drilled). The prior source-grained surface (`read_source`)
> is gone — historical notes preserved at the bottom.

## Overview

NotebookLM-style inline citations. The LLM cites sections using `[N]` tokens
embedded in prose. The backend parses these tokens server-side, emits structured
`citation` SSE events interleaved with stripped text deltas, and persists the
interleaved structure as `metadata.content_blocks`. The frontend renders inline
numbered chips that click-to-open the cited source AND select that section's
tab in the source-detail viewer (the section tabs from #454).

## Architecture

```
LLM delta stream
  → CitationParser (regex [\d+], validates against registry; gates on entry.citable)
  → SSE: interleaved "delta" + "citation" events (citation carries section_id + timestamp_start)
  → Frontend: assembles ContentBlock[] + SectionCoverage[]
  → Render: ReactMarkdown for text, CitationChip for citations (chip click → source + section tab)
  → Persist: metadata.content_blocks + metadata.rag.calls (section-grained) for history reload
```

## Key files

| File | Role |
|---|---|
| `backend/src/bibilab/pipeline/chat_tools.py` | `CitationRegistryEntry` (section-keyed, `citable` flag), `_mmss`, `_build_section_fence_header`, `_build_fenced_sections`, `_section_for_seg` containment mapper, `execute_find_passages` (section-grouped + facet→outline), `execute_read_section`, `reseed_citation_registry` |
| `backend/src/bibilab/pipeline/citation_parser.py` | `parse_delta`, `_expand_indices` — incremental regex parser; emits citation event only for citable entries |
| `backend/src/bibilab/routers/chat.py` | `SSE_EVENT_CITATION`, `stream_with_tools` registry+parser integration, `chat_endpoint` content_blocks persistence, `build_grounding_prompt` (section-grained, with #396 coverage-exemption) |
| `backend/src/bibilab/pipeline/chat_summary.py` | Compression prompt (citation preservation removed) |
| `web/src/lib/chat-utils.ts` | `ContentBlock` (citation has `section_id` + `timestamp_start`), `SectionCoverage` (replaces `RagSource`), `stripLegacyTokens` |
| `web/src/components/lists/hooks/useSSEStream.ts` | SSE consumer — accumulates `ContentBlock[]`, dispatches `find_passages` + `read_section` `tool_result` events |
| `web/src/components/lists/hooks/useConversationHistory.ts` | `MessageUI` with `contentBlocks` + `rag`, legacy source-grained fallback |
| `web/src/components/lists/ChatPanel.tsx` | `CitationChip` component (section-aware), block-based renderer |
| `web/src/components/lists/ToolLedgerRow.tsx` | `read_section` row + section coverage rendering |
| `web/src/pages/ListDetailPage.tsx` | `handleOpenSource` with section target (`sectionId` + `timestampStart` opts → opens source + selects section tab) |

## Data flow

1. User sends message → `chat_endpoint` resolves facet, builds `source_pool` + `facet_scope`
2. `stream_with_tools` starts with empty `registry: dict[section_id, CitationRegistryEntry]`
3. LLM calls `find_passages` (locator) or `read_section(section_id="[N]")` (drill) → `execute_find_passages` / `execute_read_section` assign indices, format body
4. `find_passages` tool result: chunks grouped by section under per-section `===== [N] "Title" · Section M (mm:ss–mm:ss) =====` fences; within a fence, fragments render in chronological (segment) order — not rerank order — with a `[…]` gap marker between non-seg-adjacent fragments to mark elided transcript; on facet match, full section OUTLINE (every section, summaries only, non-citable) is emitted instead
5. `read_section` tool result: one section's bounded verbatim transcript (segments within `seg_start..seg_end`), `citable=True` on success
6. LLM responds with deltas containing `[N]` tokens
7. `CitationParser` strips `[N]`, emits `citation` SSE events with `{index, section_id, source_id, timestamp_start, chunk_ids}` — **only for citable entries**; outline-only `[N]` renders as plain text
8. Frontend assembles `ContentBlock[]` + `SectionCoverage[]` from interleaved delta + citation events
9. On stream end: `content_blocks` persisted in `metadata`, `rag.calls[]` with `section_coverage` (find_passages) or `read_section` rows, ordered by registry index

## Chunk ID format

`chunk_ids` are synthetic strings `"{source_id}_{int(start_seconds)}_{int(end_seconds)}"`,
not foreign keys to `chunks` table. Built in `execute_find_passages`. Highlighting
must re-derive timestamps by parsing these strings; consider replacing with
real chunk UUIDs if and when chunks gain stable IDs in the embedding pipeline.
Only `find_passages` emits `chunk_ids`; `read_section` citations carry none
(the body is bounded verbatim, not chunked).

## Citation grain (section-level, bounded)

Each `[N]` is a **section**, not a source. A single source can surface multiple
sections this turn, each with its own `[N]`. The LLM cites a section either
by its summary/fragments (from a `find_passages` fence) or by its bounded
verbatim (from a `read_section` call).

- **One `[N]` per section.** `CitationRegistryEntry` is keyed by `section_id`
  (was `source_id` pre-#455). Each entry carries `section_id`, `source_id`,
  `seq` (1-based section index within the source), `timestamp_start`, `timestamp_end`,
  `citable` (False for outline-only, True once verbatim is shown).
- **Outline-only `[N]` are non-citable.** When `find_passages` matches a facet
  and emits the full section outline, every section gets a `[N]` with `citable=False`.
  The LLM cannot cite them; `[N]` in the response renders as plain text. After
  `read_section` drills one, `citable` flips to True and the citation event fires.
  This is the #396 coverage-confabulation fix (enforced in `citation_parser._expand_indices`,
  not prompt-only).
- **Bounded verbatim, not continuous.** `_build_section_narrative` loads only
  the section's `seg_start..seg_end` range and renders speaker-turn
  reconstruction via `format_turns`. No silent truncation, no overflow.
- **Per-section click-to-seek works.** The citation payload carries `section_id` +
  `timestamp_start` (T7); the frontend's `handleOpenSource` opts widen to
  `{sectionId, timestampStart}`; PR2 (frontend) selects the cited section's tab
  in the source-detail viewer. No `[N.M]` chunk-level scheme needed — section
  grain is fine enough.
- **Shared `[N]` counter with `find_passages`, dedup by `section_id`.** If
  `find_passages` already registered a section this turn, `read_section` reuses
  the same `[N]`. So a section located and then drilled keeps one stable number.
- **Cross-turn reseed is section-keyed.** `reseed_citation_registry` walks stored
  `tool_blocks` whose name is in `RETRIEVE_TOOL_NAMES` and rebinds their
  section-keyed `[N]`. Legacy source-keyed chunks degrade gracefully
  (`if not section_id: continue`); their `[N]` is inert text in old prose.

## Prior design (pre-#455, source-grained)

For historical reference only — the `read_source(source_id|facet)` tool is gone.
The LLM previously cited a whole source as a single `[N]`; per-chunk click-to-seek
was deferred (would have needed an `[N.M]` chunk-level scheme). The #440 fix
allowed the LLM to pass `[N]` instead of UUIDs to `read_source`, but the source
still returned a continuous transcript, so the citation remained coarse.
`#455` reworked the surface: `[N]` is now a section, and `read_section` returns
that section's bounded transcript. See PR #462 for the full migration.

## Deferred

- Hover snippet preview: lazy-load chunk text
- Cross-source comparison flow (no current user need)
