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
├── bibilab.db         SQLite (lists, jobs, sources, artifacts)
├── covers/            cached cover images
├── transcripts/       raw Whisper segments
├── artifacts/         generated artifact content
├── chroma/            ChromaDB vector data
└── downloads/         temp video files, cleaned after pipeline
```

## Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Path storage | Relative paths in DB, resolved at read time | Enables home directory migration without DB updates |
| Digest storage | Summary and keywords in `sources` table | No intermediate .md file needed |
| Overview generation | On-demand `POST /lists/:id/overview` | User controls when to generate; no silent LLM calls in pipeline |
| Job vs source dedup | `sources` is the dedup source; `jobs` is ephemeral | A video is "processed" if it has a `sources` row |
| Transcript storage | Files on disk, not in DB | Re-chunking never requires re-transcription |
| Artifact storage | Content on disk (`artifacts/{id}.md`), metadata in SQLite | Same pattern as transcripts |
| Backend serves SPA | FastAPI mounts `/assets` + catch-all → `index.html` | Single-port; no separate frontend server |
| Worker concurrency | Configurable via `config.backend.worker_concurrency` | Default 1 |
| Chat conversations | One conversation per list, auto-created on first message | `get_or_create_conversation` with `ON CONFLICT` upsert |
| Chat streaming | SSE via `StreamingResponse`; RAG context in system prompt | Citations parsed client-side from `[title @ Ts-Ts]` format |
| Conversation compression | Background summary after 30+ messages, sliding window of 20 | Keeps recent context; old messages deleted after summarization |

## Notes

- Active specs in `docs/specs/`; internal docs in `docs/internal/`; roadmap in `docs/roadmap.md`.
- Pre-commit hooks enforce ruff lint/format on backend and trailing whitespace globally. Run `pre-commit install` once after cloning.
- Config lives at `~/.bibilab/config.json`; runtime state at `~/.bibilab/`.
