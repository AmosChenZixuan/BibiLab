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

**Backend** — `backend/src/locus/`

```
routers/     — one APIRouter per module; aggregated in main.py
models/      — Pydantic request/response models
pipeline/    — one file per stage (audio → transcribe → chunk → extract → notes → embed)
adapters/    — platform-specific download + resolution
db.py        — SQLite schema + query helpers
config.py    — settings persisted to ~/.locus/config.json
worker.py    — SQLite-polling job dispatcher
```

**Web** — `web/src/`

```
components/ui/  — primitive components (Button, Modal, Panel, etc.)
components/*/   — feature components (lists/, jobs/, layout/, settings/)
pages/          — route-level page components
lib/            — typed api client, types, utilities
app/            — router, language context
```

## Conventions

### Backend

- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for classes and Pydantic models
- **Pydantic models**: `{Operation}Request` / `{Operation}Response` suffix; enums use `PascalCase` name, `UPPERCASE` values
- **Router pattern**: one `APIRouter` per file, handlers registered with explicit HTTP method decorator and `status_code=201` where appropriate
- **DB**: `sqlite3` with `asynccontextmanager` get_db wrapper; all queries use `?` placeholders, never f-string interpolation
- **Imports**: stdlib → third-party → local, with blank lines between groups
- **Errors**: `HTTPException(status_code=N, detail=...)` for HTTP errors; custom exceptions (`AuthRequiredError`, `DownloadError`, `PipelineError`) for domain errors

### Web

- **Files**: `PascalCase` for components, `kebab-case` for utilities
- **Components**: props interface above component, `ComponentPropsWithoutRef<"tag">` for root element props, `Record<Variant, string>` for variant maps
- **Handlers**: named `handle{Action}`; event props use `on{Action}` prefix (`onDelete`, `onCreate`)
- **State**: `useState` with `set` prefix; async operations use cancellation pattern via `let cancelled = false` guard in `useEffect`
- **Imports**: use `@/*` alias (`@/components/ui`, `@/lib/api`, `@/lib/types`)
- **API client**: single `api` object in `lib/api.ts` with typed `request<T>` wrapper; errors thrown as `ApiError`
- **Design tokens**: use only tokens from `web/src/styles/app.css` (`--color-*`, `--z-*`, `--font-*`); no arbitrary values

## Notes

- Technical specification, API contracts, DB schema in `docs/design-doc.md`; active specs in `docs/specs/`; plans in `docs/plans/`.
- Pre-commit hooks enforce ruff lint/format on backend and trailing whitespace globally. Run `pre-commit install` once after cloning.
- The legacy `plugin/` directory is not the active v0 interface.
- Config lives at `~/.locus/config.json`; runtime state at `~/.locus/`.
