# CLAUDE.md

## What This Project Is

**Project Bibilab** transforms video content into searchable, AI-assisted private notebooks. A FastAPI backend runs the local processing pipeline (download → transcribe → chunk → digest ∥ embed), and a React + TypeScript SPA under `web/` provides the primary user interface.

Platform-specific context lives in `backend/CLAUDE.md` and `web/CLAUDE.md`.

## Goals & Non-Goals

### Goals
- Transform individual videos and playlists into structured AI digests
- Support local transcription (Faster Whisper) and local or cloud LLMs
- Per-list RAG chat with transcript citations, streaming responses, and tool calling
- Provide on-demand list-level overview export
- Run entirely on a single user's machine

### Non-Goals
- Not a general-purpose video player
- Not a cloud or multi-user service
- Interactive timestamp seeking is not required for v0–v2
- Not building a general search engine across arbitrary content

## Architecture

```
React SPA (web/)  ↔  REST /api/*  ↔  FastAPI Backend (backend/)
```

Single-port deployment: FastAPI serves the React build as static files in production. In dev, Vite runs on `:5173` with a proxy to the backend on `:8765`.

## Storage Layout

```
~/.bibilab/
├── config.json        Pydantic settings, credentials
├── bibilab.db         SQLite (lists, jobs, sources, artifacts, conversations, messages, chunks_fts)
├── covers/            cached cover images
├── transcripts/       raw Whisper segments
├── artifacts/         generated artifact content
├── chroma/            ChromaDB vector data
├── models/            cached embedding + reranker models (downloaded on first use)
└── downloads/         temp video files, cleaned after pipeline
```

## Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Path storage | Relative paths in DB, resolved at read time | Enables home directory migration without DB updates |
| Digest storage | Summary and keywords in `sources` table | No intermediate .md file needed |
| Facet edit vs extract | `parse_facet_int`/`clean_str_facet` shared; digest path degrades bad values to `null`, manual `PATCH /sources/:id/facets` raises → 422; manual write is REPLACE (`update_source_facets`, explicit null clears) vs digest COALESCE-preserve; PATCH returns 204 | A typed edit is deliberate (reject), an LLM guess is best-effort (degrade) |
| Overview generation | On-demand `POST /lists/:id/overview` | User controls when to generate; no silent LLM calls in pipeline |
| Job vs source dedup | `sources` is the dedup source; `jobs` is ephemeral | A video is "processed" if it has a `sources` row |
| Transcript storage | Files on disk, not in DB | Re-chunking never requires re-transcription |
| Artifact storage | Content on disk (`artifacts/{id}.md`), metadata in SQLite | Same pattern as transcripts |
| Backend serves SPA | FastAPI mounts `/assets` + catch-all → `index.html` | Single-port; no separate frontend server |
| Worker concurrency | Configurable via `config.backend.worker_concurrency` | Default 1 |
| Chat conversations | One conversation per list, auto-created on first message | `get_or_create_conversation` with `ON CONFLICT` upsert |
| Conversation compression | Background summary after 30+ messages, sliding window of 10 | Summary is prose only — `[N]` citations are deliberately dropped in compression (see `docs/citation_system.md`); live citations survive only on post-window messages; old messages deleted after summarization |
| Retrieval | Deterministic facet scoping is the sole pre-retrieval narrowing (`sequence_number`/`season_number` narrow the full source pool before search; fail-open on zero-match or facet-DB-error → full pool, `facet_scope.no_match` + LLM-visible note) → Hybrid (BM25 + vector) fused via RRF (k=60), dynamic pool = max(min(sources_total×3, 10, 60), params.top_k) → cross-encoder rerank (Xenova/bge-reranker-base, zh+en) → diverse top-k with per-source depth cap | LLM dispatches `retrieve`/`survey`/`retrieve_scoped` mid-stream; `_TOOL_NAME_TO_PARAMS` maps tool name → `RetrievalParams` (mode derived from tool name: retrieve→narrow, survey→survey, retrieve_scoped→narrow); LLM *extracts* facets, backend *matches* them deterministically; non-scoped tools strip facets defensively; no source-list / `exclude_source_ids` index decision (static-per-language prompt, cacheable); `scoped_pool_size` is the full pool — see `facet_scope.matched_count`; `series_name` matching deferred |
| Tool-calling chat | `stream_with_tools` wraps `stream_llm` in a bounded loop (max 3 iterations); loopback tools (`retrieve`, `survey`, `retrieve_scoped`) feed results back for another turn; terminal tools (`generate_report`) exit; exhausted iterations force synthesis (no tools) instead of a hard error; empty synthesis triggers one forced follow-up LLM call | Eliminates the 8K-token pre-stream classifier; LLM decides which retrieval tool (if any) to use based on the question |
| Chat streaming | SSE via `StreamingResponse`; LLM dispatches tool calls mid-stream; preamble text streamed to client immediately (not buffered until stream end); any retrieve-family tool disables all three for the rest of the turn; `[N]` citations streamed as `citation` SSE events; error events carry a machine-readable `error_reason` code (e.g. `llm_rate_limit_error`) for frontend i18n | `tool_call_start`, `tool_result`, `citation`, `delta`, `rag`, `done`, `error`, `cancelled` event types; `rag` carries the final authoritative ledger before the terminal; `parse_delta` strips `[N]` markers and emits citation events with `{index, source_id, chunk_ids}` |

## Code Health Rules

Every change — feature, fix, or refactor — must leave the codebase no worse than it found it. These rules exist because past fixes introduced dead code, duplicated logic, and magic strings. Do not repeat those mistakes.

1. **No dead code.** Never commit a function, import, or variable that has zero callers. If you write a helper "for future use", delete it — add it in the PR that actually uses it.
2. **No magic strings.** Repeated string literals (localStorage keys, header names, event names) must use a shared constant. Before introducing a new literal, grep for it — if it already appears elsewhere, use or create the constant.
3. **No duplicated logic.** Before writing a utility function, check if one already exists in `lib/` (frontend) or `pipeline/` (backend). If a near-duplicate exists, extend it rather than creating a second version.
4. **No business logic in the data layer.** `db.py` is for SQL queries, not status mapping or side effects. Domain rules belong in routers or service functions.
5. **Scope discipline.** A bug fix changes only what's broken. A feature adds only what's specified. If you notice adjacent issues, file them — don't fix them in the same PR.
6. **Verify before committing.** Run `uv run pytest` (backend) and `npm test && npm run lint` (frontend) before declaring work done.

## Commit & PR Conventions

**Commit messages:** `"<type> | <scope> | #<issue> <description>"`

- Type: `feat`, `fix`, `refactor`, `chore`, `docs`
- Issue number optional for non-issue commits

**PR titles:** same as commit title, prefix with `#<issue>` when applicable

## Notes

- Active specs in `docs/specs/`; internal docs in `docs/internal/`; roadmap in `docs/roadmap.md`.
- Pre-commit hooks enforce ruff lint/format on backend and trailing whitespace globally. Run `pre-commit install` once after cloning.
- Config lives at `~/.bibilab/config.json`; runtime state at `~/.bibilab/`.
