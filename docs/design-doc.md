# Project Locus — Technical Design Document

> Version: 0.6 (Draft)
> Last updated: 2026-03-31

---

## 1. Overview

**Project Locus** is a local, self-hosted video knowledge base. It ingests videos from supported platforms, transcribes them locally, extracts structured knowledge using an LLM, and surfaces everything through a purpose-built web UI — a local alternative to NotebookLM with full model selection.

**One-line pitch:** Turn any video playlist into a searchable, AI-queryable knowledge base — entirely on your own machine, with your own models.

---

## 2. Goals & Non-Goals

### Goals
- Process individual videos, playlists, and courses into structured markdown notes
- Support local-first transcription (Faster Whisper) and local or cloud LLMs
- Enable AI Q&A grounded in your own video corpus, with transcript citations (v1)
- Provide a list-level overview that can be manually generated and downloaded
- Progressively enhance with multimodal and generative features in later versions

### Non-Goals
- Not a general-purpose video player
- Not a cloud service — designed for single-user local deployment (multi-user is a v-future concern)
- Interactive timestamp seeking is not required for v0–v2 (timestamps appear as plain text references)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│              Locus Web UI (React + TypeScript)       │
│                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │  Sources     │  │    Chat    │  │   Studio    │ │
│  │  (ingestion) │  │  (v1 RAG)  │  │  (overview) │ │
│  └──────────────┘  └────────────┘  └─────────────┘ │
└───────────────────────│─────────────────────────────┘
                        │ HTTP (REST + SSE)
┌───────────────────────▼─────────────────────────────┐
│                  Locus Backend (Python)              │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │  Ingest  │  │ Process  │  │     Q&A / RAG     │ │
│  │  Router  │  │ Pipeline │  │     Engine (v1)   │ │
│  └────┬─────┘  └────┬─────┘  └─────────┬─────────┘ │
│       │             │                  │            │
│  ┌────▼─────────────▼──────────────────▼─────────┐ │
│  │               Job Queue (SQLite)               │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ Platform     │  │  Whisper   │  │ Vector Store│ │
│  │ Adapters     │  │ (local)    │  │ (ChromaDB)  │ │
│  └──────────────┘  └────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
       Bilibili      YouTube        (v3+)
       Adapter       Adapter
```

---

## 4. Component Breakdown

### 4.1 Web UI (React + TypeScript)

A single-page application served by the FastAPI backend at `localhost:8765`.

**Pages:**

- `/` — Home: grid of Lists, create new list
- `/lists/:id` — List detail: three-column layout (Sources | Chat | Studio)
- `/settings` — Global config: AI provider, Whisper, accounts, health panel

**List detail layout:**

```
┌──────────────────┬──────────────────┬──────────────────┐
│   Sources         │      Chat        │     Studio       │
│                   │                  │                  │
│ [Add URL input]   │  (v0: skeleton,  │ [Generate        │
│                   │   v1: RAG chat)  │  Overview btn]   │
│ ── Source A ──    │                  │                  │
│ ── Source B ──    │                  │ ── Artifact ──   │
│ ── Source C ──    │                  │ overview.md      │
│                   │                  │ [Download]       │
└──────────────────┴──────────────────┴──────────────────┘
```

Clicking a source replaces the entire left (Sources) panel with a detail view:
- `← Back` returns to source list
- **Note** tab: read-only LLM-generated note
- **Transcript** tab: raw Whisper transcript

Notes are **read-only** in the UI. The only export action is download as markdown.

**Jobs:** Global floating badge (bottom-right) showing active job count. Clicking opens a side drawer with job list, status badges, progress bars, and cancel buttons. Auto-polls every 3s when open.

**Dev setup:** Vite dev server on `:5173`, proxies `/api/*` to `:8765`. Windows browser reaches both directly over WSL network. Production: `npm run build` → `web/dist`, FastAPI mounts as static files after all `/api` routes.

### 4.2 Backend API (Python / FastAPI)

Core routes:
```
POST   /ingest/url              # body: { list_id, url } — single video, playlist, or course
                                # ?rerun=true overrides deduplication and re-runs the full pipeline
GET    /jobs                    # list all jobs with status
GET    /jobs/{id}               # single job status + progress
DELETE /jobs/{id}               # cancel job

POST   /lists                   # create a new list
GET    /lists                   # all lists
DELETE /lists/{id}              # delete list and cascade: notes + processing_log + embeddings
GET    /lists/{id}/sources      # sources (processing_log rows) for a list
DELETE /lists/{id}/sources/{video_id}  # delete one source: note file + processing_log + embeddings
POST   /lists/{id}/overview     # on-demand: generate overview markdown, return for download

GET    /notes/{video_id}/content    # note markdown content
GET    /notes/{video_id}/transcript # raw transcript text

GET    /models/whisper          # list available Whisper models with download status
POST   /models/whisper/download # queue a Whisper model download job

GET    /health                  # dependency health check
GET    /config                  # get config
PUT    /config                  # update config

# v1
POST   /lists/{id}/query        # SSE: multi-turn RAG Q&A scoped to a list
```

**Ingestion flow:** The user creates a list first (or selects an existing one), then submits a URL into it. Ingestion without a list is not permitted.

**Deduplication:** Before queuing a video, the backend checks the `sources` table for an existing `video_id`. If found, the job is skipped silently.

`DELETE /lists/{id}/sources/{video_id}` removes the note file, ChromaDB embeddings, and the `sources` row. After deletion, re-ingesting the same URL is treated as a fresh video.

`POST /ingest/url?rerun=true` bypasses the deduplication check and re-runs the full pipeline in-place, overwriting the existing note and embeddings (use when re-processing with an upgraded model without deleting first).

### 4.3 Job Queue, Lists & Sources

SQLite serves three purposes: a transient job queue, a list registry, and a source catalog.

**Lists are stored in SQLite** as the backend's source of truth.

**Job states:** `queued → downloading → transcribing → extracting → writing → done | failed`

Jobs are ephemeral — they can be pruned after a retention window once complete. `sources` is the permanent catalog.

```sql
-- List registry
CREATE TABLE lists (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Ephemeral queue: pruned after retention window
CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT,           -- 'video' | 'playlist' | 'course'
    source_url  TEXT,
    platform    TEXT,
    status      TEXT,           -- queued | downloading | transcribing | extracting | writing | done | failed | needs_auth
    progress    INTEGER,        -- 0-100
    error       TEXT,
    created_at  DATETIME,
    updated_at  DATETIME,
    meta        JSON            -- platform-specific metadata
);

-- Active source catalog: one row per successfully processed video, deleted when source is removed
CREATE TABLE sources (
    video_id            TEXT PRIMARY KEY,   -- platform-native ID (e.g. bvid)
    platform            TEXT,
    list_id             TEXT REFERENCES lists(id),
    note_path           TEXT NOT NULL,      -- absolute path to ~/.locus/notes/{video_id}.md
    transcript_path     TEXT,               -- absolute path to ~/.locus/transcripts/{video_id}.txt
    whisper_model       TEXT,
    ai_model            TEXT,
    vision_enabled      BOOLEAN,
    processed_at        DATETIME,
    settings_snapshot   JSON
);
```

`sources` powers deduplication (is this video already processed?), source listing per list, and note path resolution.

### 4.4 Platform Adapters

Each adapter implements a common interface:

```python
class PlatformAdapter:
    def resolve(self, url: str) -> VideoMeta | PlaylistMeta
    def download(self, video_id: str, session: Session | None) -> Path
    def requires_auth(self, resource_type: str) -> bool
```

**v0:** Bilibili adapter (single video `bvid`, playlist, course)
**v3:** YouTube adapter, with free-text resolver

Auth is handled per-adapter. Bilibili uses cookie-based session stored in config. When a download returns a 403, the adapter raises `AuthRequiredError` and the job transitions to `needs_auth`, prompting the user via the UI.

### 4.5 Processing Pipeline

Per video, once downloaded:

```
1. Extract audio (FFmpeg)
2. Transcribe audio → raw Whisper segments (Faster Whisper)
3. Write raw segments to transcript file (~/.locus/transcripts/{video_id}.txt)
4. Merge consecutive segments into RAG chunks (~200-400 tokens each)
5. LLM pass: extract title, summary, key points with timestamps
6. (v1) If vision enabled: sample frames → multimodal LLM pass
7. Write video note (~/.locus/notes/{video_id}.md)
8. Embed RAG chunks into ChromaDB vector store
9. Write sources entry
```

Note: List overview is **not** generated automatically during pipeline. It is generated on-demand via `POST /lists/{id}/overview` from the Studio panel.

**Chunking strategy:** Whisper raw segments (~5–15s each) are merged greedily until the chunk reaches ~300 tokens. Each chunk stores `timestamp_start`, `timestamp_end`, and `sequence_index` in ChromaDB metadata.

### 4.6 Vector Store (ChromaDB)

Used for RAG in the Q&A engine (v1). Each chunk stored with metadata:

```json
{
  "video_id": "...",
  "list_id": "...",
  "video_title": "...",
  "timestamp_start": 142,
  "timestamp_end": 198,
  "sequence_index": 7,
  "text": "..."
}
```

Q&A queries are scoped by `list_id` (list-level) or `video_id` (single-source).

### 4.7 Transcript Storage & Serving

Raw Whisper segments stored at `~/.locus/transcripts/{video_id}.txt`. Served via:

```
GET /notes/{video_id}/transcript
```

The web UI renders transcript in the Source viewer's transcript tab (read-only).

### 4.8 Free-Text Ingestion Resolver (v3)

When the user submits a natural language query, the backend:

1. Sends the query to the LLM to extract intent: platform hint, search terms, content type
2. Dispatches to the matching platform's search skill
3. Returns candidate playlists/videos for user confirmation before ingesting
4. On confirmation, creates jobs as normal

Always requires user confirmation before bulk ingestion.

---

## 5. Data Model — Notes

### 5.1 Storage Layout

Notes and transcripts are stored under `~/.locus/`.

```
~/.locus/
  locus.db                  # SQLite (lists + jobs + processing_log)
  notes/
    {video_id}.md           # LLM-generated note
    attachments/
      {video_id}_cover.jpg
  transcripts/
    {video_id}.txt          # raw Whisper segments: [HH:MM:SS] text
  chroma/                   # ChromaDB data directory
  downloads/                # temporary video files, cleaned up after processing
```

### 5.2 Video Note

Stored at: `~/.locus/notes/{video_id}.md`

```markdown
---
video_id: bv1abc123
platform: bilibili
source_url: https://www.bilibili.com/video/BV1abc123
list_id: list_xyz
duration: 3842
processed_at: 2026-03-31T10:00:00
---

# {Video Title}

![cover](attachments/bv1abc123_cover.jpg)

## Summary
{LLM-generated summary}

## Key Points
- [00:02:14] {knowledge point}
- [00:08:45] {knowledge point}
- [00:21:03] {knowledge point}
```

### 5.3 List Overview

Generated on-demand via `POST /lists/{id}/overview`. Returned as a markdown string in the API response body; the client triggers a browser download. Not stored on the filesystem.

```markdown
# {List Name} — Overview

## Outline
{LLM-generated outline synthesizing all sources in the list}

## Sources
- {Video Title 1}
- {Video Title 2}
...
```

---

## 6. Configuration Schema

```json
{
  "accounts": {
    "bilibili": {
      "cookie": "...",
      "last_verified": "..."
    }
  },
  "ai": {
    "provider": "openai | anthropic | ollama | custom",
    "model": "gpt-4o",
    "api_key": "...",
    "base_url": null
  },
  "transcription": {
    "engine": "faster-whisper",
    "model_size": "large-v3",
    "device": "cuda | cpu",
    "language": "auto"
  },
  "vision": {
    "enabled": false,
    "frame_sample_rate": 30,
    "model": null
  },
  "backend": {
    "port": 8765,
    "worker_concurrency": 1
  }
}
```

---

## 7. Deployment Health Checks

The `/health` endpoint checks all dependencies. Displayed inline in the Settings page.

| Dependency | Type | Blocks ingestion? |
|---|---|---|
| Backend API | self | yes |
| LLM API / Ollama | external | yes |
| Faster Whisper model | local | yes |
| FFmpeg binary | local | yes |
| Embedding model (ONNX) | local | yes |
| CUDA availability | local | no (falls back to CPU) |
| Bilibili session | account | no (blocks course only) |

---

## 8. Versioned Rollout

### v0 — Core Pipeline + Web UI
- Global config + deployment health
- Bilibili ingestion: single video, playlist, course
- Faster Whisper transcription
- LLM-based note generation (summary + timestamped key points)
- Markdown note storage at `~/.locus/notes/`
- List creation and management (DB-backed)
- On-demand list overview generation (manual, Studio panel)
- Job queue with status polling in web UI
- Source viewer: read-only note + transcript toggle

### v1 — Enhanced Knowledge
- Multimodal vision pass (opt-in, configurable frame sampling)
- Multi-turn RAG Q&A (list scope) with transcript citation and timestamp references
- Source truth panel: user-supplied corrections injected into RAG context

### v2 — Generative Outputs
- TTS configuration
- Mindmap generation (Mermaid)
- Audio overview (LLM script + TTS, scoped to list)

### v3 — Expanded Sources
- YouTube adapter
- Free-text ingestion resolver with per-platform search skills
- User confirmation step before bulk ingestion from free-text

---

## 9. Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frontend | React + TypeScript SPA | Full control over UX for the product's three-panel interface, and deployable without SSR |
| Serving | FastAPI serves React build as static files | Single port, no separate frontend server in production |
| Note storage | Notes stored in `~/.locus/notes/` | Keeps note generation, serving, and download behavior fully under backend control |
| List storage | SQLite `lists` table | Database-backed registry is the natural source of truth for list lifecycle and routing |
| Overview generation | On-demand via API, not in pipeline | Avoids silent LLM calls during ingestion; user controls when to generate and download |
| Backend framework | FastAPI | Async-native, easy SSE for job progress and chat streaming |
| DB schema | SQLite three tables: `lists`, `jobs`, `sources` | `jobs` is ephemeral queue (prunable); `sources` is the active catalog (dedup + note path); `lists` is the registry. Naming reflects purpose, not implementation. |
| Vector store | ChromaDB | Local, no server process, Python-native |
| Transcription | Faster Whisper | Best local quality/speed tradeoff, CUDA support |
| Transcript storage | `~/.locus/transcripts/` | Decoupled from notes; re-chunking/re-embedding never requires re-transcription |
| RAG chunking | Greedy merge of Whisper segments (~300 tokens) | Balances embedding quality with timestamp granularity |
| Note format | Markdown + YAML frontmatter | Human-readable, downloadable, future-portable |
| Auth storage | Local config file | Credentials stay off network |

---

## 10. Open Questions

1. ~~**Whisper language detection**~~ — Resolved: UI dropdown with `auto / zh / en`. Stored in config, applied globally.
2. ~~**Note deduplication**~~ — Resolved: check `sources` table for existing `video_id`. Skip silently if found. Override with `?rerun=true` (re-process in-place) or delete the source first (re-add fresh).
3. ~~**List assignment**~~ — Resolved: user always selects a list before ingesting. No default list.
4. ~~**Note sync model**~~ — Resolved: notes are managed by the backend; the web UI is read-only.
5. ~~**Chunk strategy for RAG**~~ — Resolved: greedy merge of Whisper segments to ~300 token target.
6. ~~**List storage**~~ — Resolved: DB-backed (`lists` table in SQLite).
7. ~~**Frontend approach**~~ — Resolved: React + TypeScript SPA served by FastAPI.
8. ~~**processing_log vs jobs**~~ — Resolved: `processing_log` renamed to `sources` (active catalog, mutable). `jobs` stays ephemeral. "Log" naming was misleading — it implied immutability, but the table is a catalog of active sources, not an audit trail.
