# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Project Locus** transforms video content (Bilibili, eventually YouTube) into searchable, AI-queryable Obsidian notes. A FastAPI backend runs a local processing pipeline (download → transcribe → chunk → extract → notes → embed), while an Obsidian TypeScript plugin provides the user interface.

## Commands

### Backend (Python, managed with `uv`)

```bash
cd backend
uv sync --dev            # Install all dependencies
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest            # All tests
uv run pytest tests/test_ingest.py -v  # Single test file
uv run python -m locus.main            # Start server (localhost:8765)
```

### Plugin (TypeScript, managed with `npm`)

```bash
cd plugin
npm install      # Install dependencies
npm run dev      # Watch mode (esbuild)
npm run build    # Production build (type check + bundle)
npm run lint     # ESLint
```

## Architecture

```
Obsidian Plugin (TypeScript)
        ↕ REST/HTTP (localhost:8765)
FastAPI Backend (Python)
    ├── Job Queue (SQLite polling, worker.py)
    ├── Platform Adapters (adapters/bilibili.py)
    ├── Processing Pipeline (pipeline/)
    │   audio → transcribe → chunk → extract → notes → embed
    └── Storage
        ├── SQLite  — job queue + processing log
        ├── ChromaDB — vector embeddings (local, embedded)
        └── ~/.locus/ — transcripts, downloads, chroma data
```

**Key design decisions:**
- **Two-table DB strategy:** `jobs` (ephemeral queue) + `processing_log` (permanent history). Jobs restart-safe by re-queueing `queued`/`in_progress` rows on startup.
- **RAG chunking:** Greedy merge of Whisper segments to ~300 tokens. Source transcript never modified; stored at `~/.locus/transcripts/` (outside vault, invisible to Obsidian by default).
- **Local-first:** No cloud services required. ChromaDB runs embedded, transcription via Faster Whisper (CUDA-capable), video download via yt-dlp.
- **Platform adapters:** `adapters/base.py` defines the `PlatformAdapter` ABC; `bilibili.py` is the only current implementation.

## Code Layout

```
backend/src/locus/
├── main.py           # FastAPI app creation, lifespan, route registration
├── worker.py         # WorkerLoop — polls SQLite, dispatches pipeline jobs
├── db.py             # Schema bootstrap, query helpers
├── config.py         # Pydantic settings, persisted to ~/.locus/config.json
├── adapters/         # Platform-specific download + resolution logic
├── pipeline/         # One file per pipeline stage (audio, transcribe, chunk, extract, notes, embed)
├── routers/          # FastAPI route handlers (health, config, ingest, jobs, lists)
└── models/           # Pydantic request/response models

plugin/src/
└── main.ts           # Obsidian plugin entry point (onload/onunload)
```

## Notes

- Full technical specification (API routes, DB schema, data models, config schema, rollout roadmap) lives in `docs/design-doc.md`.
- Pre-commit hooks enforce ruff lint/format on backend and trailing whitespace globally. Run `pre-commit install` once after cloning.
- The plugin build artifact `plugin/main.js` is committed to git (Obsidian plugin convention).
- Config is stored at `~/.locus/config.json`; runtime state at `~/.locus/` (db, transcripts, chromadb).
