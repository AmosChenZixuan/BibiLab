# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Project Locus** transforms video content into searchable, AI-assisted private notebooks. A FastAPI backend runs the local processing pipeline (download → transcribe → chunk → extract → notes → embed), and a React + TypeScript SPA under `web/` provides the primary user interface for lists, ingestion, jobs, notes, transcripts, overview export, and settings.

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

### Web UI (React + TypeScript, managed with `npm`)

```bash
cd web
npm install          # Install frontend dependencies
npm run dev          # Start Vite dev server on :5173
npm run build        # Production build to web/dist
npm run test         # Frontend test suite
npm run test -- list-detail-page   # Focused frontend tests
npm run lint         # Type-check the frontend
```

## Architecture

```
React SPA (web/, Vite in dev, static files in prod)
        ↕ REST/HTTP (/api in dev, localhost:8765 in prod)
FastAPI Backend (Python)
    ├── Job Queue (SQLite polling, worker.py)
    ├── Platform Adapters (adapters/bilibili.py)
    ├── Processing Pipeline (pipeline/)
    │   audio → transcribe → chunk → extract → notes → embed
    └── Storage
        ├── SQLite  — job queue + processing log
        ├── ChromaDB — vector embeddings (local, embedded)
        └── ~/.locus/ — config, notes, transcripts, downloads, chroma data
```

## Code Layout

```
backend/src/locus/
├── main.py           # FastAPI app creation, lifespan, route registration
├── worker.py         # WorkerLoop — polls SQLite, dispatches pipeline jobs
├── db.py             # Schema bootstrap, query helpers
├── config.py         # Pydantic settings, persisted to ~/.locus/config.json
├── adapters/         # Platform-specific download + resolution logic
├── pipeline/         # One file per pipeline stage (audio, transcribe, chunk, extract, notes, embed)
├── routers/          # FastAPI route handlers (health, config, ingest, jobs, lists, notes, transcripts, whisper)
└── models/           # Pydantic request/response models

web/src/
├── app/              # Router and shared app shell
├── components/       # Route-level UI sections (lists, sources, jobs, settings, studio)
│   ├── layout/       # App shell components (AppFrame, IdentityPanel)
│   └── ui/           # Shared primitives: Button, Input, Select, FormField, SettingsField, Panel, PanelTitle, PanelBody, StatusChip, Modal, ContextMenu, Spinner
├── pages/            # Home, list detail, settings
├── lib/              # Typed API wrappers, downloads, shared types
└── test/             # Vitest + RTL coverage
```

## Frontend Design System

Tokens are defined in `web/src/styles/app.css` (`@theme` block). Use token utility classes — never arbitrary color, shadow, or radius values.

Reusable components live in `web/src/components/ui/`.

## Notes

- Full technical specification (API routes, DB schema, data models, config schema, rollout roadmap) lives in `docs/design-doc.md`.
- Pre-commit hooks enforce ruff lint/format on backend and trailing whitespace globally. Run `pre-commit install` once after cloning.
- The legacy `plugin/` directory remains in the repo but is not the active v0 interface.
- Config is stored at `~/.locus/config.json`; runtime state at `~/.locus/` (db, notes, transcripts, downloads, chromadb, models).
