# Project Bibilab — Technical Design

> Version: 0.9
> Last updated: 2026-04-06

---

## 1. What This Is

**Project Bibilab** transforms video content into searchable, AI-assisted private notebooks. A FastAPI backend runs the local processing pipeline (download → transcribe → chunk → digest ∥ embed), and a React + TypeScript SPA provides the primary user interface.

The web UI is the product. The backend exists to serve it.

---

## 2. Goals & Non-Goals

### Goals
- Transform individual videos and playlists into structured AI digests
- Support local transcription (Faster Whisper) and local or cloud LLMs
- Enable AI Q&A grounded in the video corpus with transcript citations (v1)
- Provide on-demand list-level overview export
- Run entirely on a single user's machine

### Non-Goals
- Not a general-purpose video player
- Not a cloud or multi-user service
- Interactive timestamp seeking is not required for v0–v2 (timestamps appear as text references in digests)
- Not building a general search engine across arbitrary content

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Bibilab Web UI (React + TypeScript SPA)          │
│                                                              │
│   /               Home: grid of lists                        │
│   /lists/:id      List detail: Sources | Chat | Lab          │
│   /settings       Global config, health, accounts             │
└───────────────────────────│──────────────────────────────────┘
                            │ HTTP /api/*
┌───────────────────────────▼──────────────────────────────────┐
│                   Bibilab Backend (Python/FastAPI)               │
│                                                              │
│   Job Queue (SQLite) ──► WorkerLoop ──► Pipeline stages        │
│                                                              │
│   ┌──────────────┐  ┌────────────────────┐  ┌──────────────┐   │
│   │ Platform     │  │  Pipeline stages  │  │ Vector Store │   │
│   │ Adapters     │  │  (audio,          │  │ (ChromaDB)   │   │
│   │ (Bilibili)   │  │   transcribe,     │  │              │   │
│   │              │  │   chunk, digest,  │  │              │   │
│   │              │  │   embed)          │  └──────────────┘   │
│   └──────────────┘  └────────────────────┘                     │
│                                                              │
│   Storage: ~/.bibilab/                                          │
│     bibilab.db         SQLite (lists, jobs, sources, artifacts) │
│     covers/          thumbnail images                          │
│     transcripts/     raw Whisper output                       │
│     chroma/          ChromaDB data                            │
│     downloads/       temp video files, cleaned after process  │
└──────────────────────────────────────────────────────────────┘
```

**Serving:** FastAPI serves the React build as static files in production (`/assets`, SPA catch-all). In dev, Vite runs on `:5173` with a proxy to the backend on `:8765`.

---

## 4. Storage Layout

```
~/.bibilab/
├── config.json          Pydantic settings, credentials
├── bibilab.db             SQLite
├── covers/
│   └── {video_id}.jpg   cached cover image
├── transcripts/
│   └── {video_id}.txt   raw Whisper segments: [HH:MM:SS] text
├── artifacts/           generated artifact content
├── chroma/              ChromaDB data directory
└── downloads/           temp video files, cleaned after pipeline
```

Transcripts and covers are **always written to disk**, not stored in SQLite. `sources` holds paths and denormalized metadata (summary, keywords) for fast listing.

---

## 5. Database Schema

Four tables:

### `lists` — List registry

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `name` | User-visible name |
| `thumbnail_source_id` | FK to `sources.id` (UUID), nullable |
| `created_at` | ISO timestamp |

### `jobs` — Ephemeral queue

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `type` | `"ingest"` \| `"playlist"` \| `"course"` \| `"model_download"` \| `"artifact"` |
| `status` | `queued` → `downloading` → `transcribing` → `processing` → `done` \| `failed` \| `needs_auth` |
| `progress` | 0–100 |
| `error` | Error message, nullable |
| `meta` | JSON blob: `{ list_id, video_id, title, cover_url, platform, source_url, rerun, ... }` |

`jobs` is a transient queue — prune-able after completion. It is **not** the source of truth about what videos exist.

### `sources` — Active video catalog

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `video_id` | Platform-native ID (e.g. `bvid`) |
| `platform` | Platform name |
| `list_id` | FK to `lists.id` |
| `title` | From platform metadata (e.g. Bilibili) — enables list-level overview without reading files |
| `summary` | Denormalized from LLM output — feeds `POST /lists/:id/overview` |
| `keywords` | JSON list of keywords extracted by LLM |
| `language` | Video language code |
| `uploader` | Video uploader name |
| `duration_seconds` | Video duration in seconds |
| `transcript_path` | Relative path from `~/.bibilab/`, e.g. `transcripts/{video_id}.txt`, nullable |
| `whisper_model` | Model used for transcription |
| `ai_model` | Model used for extraction |
| `vision_enabled` | Boolean |
| `cover_url` | Remote cover URL at ingest time |
| `processed_at` | ISO timestamp |
| `settings_snapshot` | JSON blob of config at ingest time |

`sources` is the authoritative record of processed videos. It drives deduplication, listing, and digest resolution.

### `artifacts` — Lab-generated content

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `list_id` | FK to `lists.id` |
| `name` | User-visible name (initial = type label, renameable) |
| `type` | `"brief"` \| `"study_guide"` \| `"blog_post"` \| `"custom_report"` |
| `prompt` | Exact prompt string submitted |
| `source_ids` | JSON array of source UUIDs |
| `status` | `generating` → `done` \| `failed` |
| `content_path` | Relative path from `~/.bibilab/`, e.g. `artifacts/{id}.md`, nullable |
| `error` | Error message, nullable |
| `created_at` | ISO timestamp |

Artifacts use job type `"artifact"`. Content is stored on disk; metadata and status live in SQLite.

---

## 6. Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Path storage | Relative paths in DB, resolved at read time | Enables home directory migration without DB updates |
| Digest storage | Summary and keywords stored in `sources` table | Decouples digest lifecycle from DB; no intermediate .md file needed |
| List storage | SQLite `lists` table | Natural source of truth for routing and list-level queries (overview, source count, ordering) |
| Overview generation | On-demand via `POST /lists/:id/overview`, not in pipeline | Avoids silent LLM calls during ingestion; user controls when to generate |
| Job vs source deduplication | `sources` is the dedup source; `jobs` is purely ephemeral | A video is "processed" if it has a `sources` row. Re-processing requires `?rerun=true` or explicit delete |
| Overview reads `sources.title/summary` | Denormalized into `sources` at ingest time | Enables list-level synthesis without re-reading transcript files |
| Transcript storage | Files in `~/.bibilab/transcripts/`, not in DB | Re-chunking or re-embedding never requires re-transcription |
| `sources` table, not `processing_log` | Renamed from `processing_log` | The table is an active catalog (mutable), not an audit log (immutable). The old name was misleading. |
| Backend serves SPA | FastAPI mounts `/assets` + catch-all `/{path}` → `index.html` | Single-port deployment; no separate frontend server needed |
| Thumbnail auto-assign | First ingested source in a list automatically becomes the thumbnail | Avoids empty thumbnails on new lists; reassignable via settings |
| SPA routing | Client-side (React Router) | FastAPI `/{full_path:path}` with `not_found` guard on `api/*` |
| Worker concurrency | Configurable via `config.backend.worker_concurrency` | Default 1; users with GPU headroom can increase |
| ChromaDB chunk metadata | `video_id`, `list_id`, `timestamp_start/end`, `sequence_index` | Enables both per-video and per-list RAG scope |
| Artifact storage | Content on disk (`artifacts/{id}.md`), metadata in SQLite | Same pattern as transcripts; avoids large text blobs in DB |
| Lab panel naming | "Lab" (not "Studio") | Clearer metaphor for tool-based content generation |

---

## 7. Ingestion Flow

```
User submits URL → POST /ingest/url
                        │
                    BilibiliAdapter.resolve(url)
                        │
              Returns VideoMeta or PlaylistMeta
                        │
              Check already_done via sources table
                        │
              For each unprocessed video:
                  create_job(type=ingest, meta={...})
                        │
              WorkerLoop picks up job
                        │
              adapter.download(video_id) → temp file
                        │
              pipeline: audio → transcribe → chunk → digest ∥ embed
                        │
              write_source() — upserts into sources table
                        │
              Temp download file deleted
```

**Deduplication:** Checked at ingest time against `sources.video_id`. If found and `rerun=false` (default): skip silently, report in response. With `rerun=true`: re-runs full pipeline, overwrites digest and embeddings in-place.

**Delete flow:** `DELETE /lists/:id/sources/:video_id` removes the transcript file, ChromaDB embeddings, and the `sources` row. After deletion, the same URL can be re-ingested as a fresh video.

---

## 8. Processing Pipeline

Per video, in order:

```
1. download        → temp video file
2. audio           → strip audio to .wav via FFmpeg
3. transcribe      → Faster Whisper → raw segments with timestamps
4. chunk           → merge segments into RAG-ready chunks (~300 tokens)
5. digest          → LLM: title, summary, keywords with timestamps (parallel with embed)
6. embed           → store chunks in ChromaDB with metadata (parallel with digest)
7. write_source    → upsert row into sources table
```

Note: List overview (`POST /lists/:id/overview`) is **not** in the pipeline — it reads `sources.title/summary` for all sources in a list and synthesizes on-demand.

### Chunking strategy

Whisper segments are ~5–15s each. Consecutive segments are merged greedily until the chunk reaches ~300 tokens. Each chunk stores `timestamp_start`, `timestamp_end`, and `sequence_index` in ChromaDB metadata. This balances embedding quality with timestamp granularity — large enough for context, small enough to keep citations precise.

### DigestResult

The digest stage produces a structured result:

```python
class DigestResult(BaseModel):
    summary: str
    keywords: list[str]  # extracted by LLM
```

`summary` and `keywords` are denormalized into `sources.summary` and `sources.keywords`. `title` is sourced from platform metadata (e.g. Bilibili video title) and stored directly.

---

## 9. Platform Adapters

```python
class PlatformAdapter:
    def resolve(self, url: str) -> VideoMeta | PlaylistMeta
    def download(self, video_id: str, session: httpx.Session | None) -> Path
    def requires_auth(self, resource_type: str) -> bool
```

**v0:** `BilibiliAdapter` — single video (`bvid`), playlist, course. Cookie-based session auth stored in config.

When a download returns 403: adapter raises `AuthRequiredError`, job transitions to `needs_auth`, UI prompts the user.

**v3:** YouTube adapter with free-text resolver.

---

## 10. List Detail Page — Sources Panel

The Sources panel operates in two modes, toggled by user action:

**List mode** — URL input + source rows:
- Text input accepts a Bilibili URL
- Job progress row shown for in-flight ingestions (animated stage indicator)
- Source rows show title + platform; hover reveals context menu (Re-run, Delete)
- Clicking a source row opens viewer mode

**Viewer mode** — back button + content:
- Back button returns to list mode
- **Digest** tab: renders the LLM-generated summary and keywords
- **Transcript** tab: renders raw Whisper transcript in monospace, read-only

---

## 11. Configuration Schema

```json
{
  "accounts": {
    "bilibili": { "cookie": "", "last_verified": "" }
  },
  "ai": {
    "provider": "openai | anthropic | ollama | custom",
    "model": "gpt-4o",
    "api_key": "",
    "base_url": null,
    "output_language": "ui"
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

## 12. Open Questions

1. ~~Whisper language detection~~ — UI dropdown (`auto / zh / en`), stored in config, applied globally.
2. ~~Note deduplication~~ — `sources` table as dedup source; `?rerun=true` to re-process in-place.
3. ~~List assignment~~ — User always selects a list before ingesting.
4. ~~Digest sync model~~ — Backend writes digest to sources table; web UI is read-only.
5. ~~Chunk strategy~~ — Greedy merge of Whisper segments to ~300 token target.
6. ~~List storage~~ — SQLite `lists` table.
7. ~~Frontend approach~~ — React + TypeScript SPA served by FastAPI.
8. ~~processing_log naming~~ — Renamed to `sources`; table is a mutable catalog, not an immutable log.
9. **v1 RAG Q&A** — List-scoped multi-turn chat with transcript citation and timestamp references. Open: how to scope citations (per-chunk or per-turn)? How to handle multi-source ranking?
10. **v1 Multimodal vision** — Opt-in frame sampling pass. Open: which multimodal model? Does this run in pipeline or on-demand like overview?
11. **v1 Source truth panel** — User-supplied corrections injected into RAG context. Open: stored as annotations on the note file, or a separate overlay table?
12. **v2 Mindmap generation** — Mermaid output from LLM. Open: generated on-demand or stored alongside digests?
13. **v2 Audio overview** — LLM script + TTS, scoped to list. Open: TTS engine choice (local or cloud)? Does this become a downloadable artifact or a playable inline player?
14. **v3 YouTube adapter** — Adapter interface is already defined; YouTube-specific resolver and downloader are not implemented. Open: OAuth vs API key vs cookie auth?
15. **v3 Free-text resolver** — Natural language → platform search → user confirmation → bulk ingest. Open: LLM for intent extraction vs heuristic? How to handle ambiguous queries?
