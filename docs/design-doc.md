# Project Locus — Technical Design Document

> Version: 0.5 (Draft)
> Last updated: 2026-03-30

---

## 1. Overview

**Project Locus** is a personal video knowledge base tool. It ingests videos from supported platforms, transcribes them locally, extracts structured knowledge using an LLM, and surfaces everything through Obsidian — an already-deployed local tool with a rich plugin ecosystem.

**One-line pitch:** Turn any video playlist into a searchable, AI-queryable set of Obsidian notes — entirely on your own machine.

---

## 2. Goals & Non-Goals

### Goals
- Process individual videos, playlists, and courses into structured Obsidian notes
- Support local-first transcription (Faster Whisper) and local or cloud LLMs
- Enable AI Q&A grounded in your own video corpus, with transcript citations
- Provide a list-level overview note that aggregates knowledge across videos in a series
- Progressively enhance with multimodal and generative features in later versions

### Non-Goals
- Not a general-purpose video player
- Not a cloud service — designed for single-user local deployment
- Interactive timestamp seeking is not required for v0–v2 (timestamps appear as plain text references)
- Not a replacement for Obsidian's own note-taking — Locus generates notes, the user manages them

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Obsidian                         │
│  ┌─────────────────────────────────────────────┐   │
│  │           Locus Obsidian Plugin             │   │
│  │  - Ingestion UI (URL / free-text)           │   │
│  │  - Job status panel                         │   │
│  │  - AI Q&A panel (per-note, per-list)        │   │
│  │  - Config panel                             │   │
│  └────────────────────┬────────────────────────┘   │
└───────────────────────│─────────────────────────────┘
                        │ HTTP (REST)
┌───────────────────────▼─────────────────────────────┐
│                  Locus Backend (Python)              │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │  Ingest  │  │ Process  │  │     Q&A / RAG     │ │
│  │  Router  │  │ Pipeline │  │     Engine        │ │
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

### 4.1 Obsidian Plugin

Responsibilities:
- Render the ingestion form (URL input or free-text query)
- Poll job status and display progress
- Render the AI Q&A chat panel (per-note and per-list scope)
- Host the global config UI (accounts, models, deployment health)

Communication: All plugin-to-backend calls are REST over `localhost`. The backend port is configurable (default: `8765`).

**Vault sync (v0):** The plugin registers two vault event listeners on load:

- `vault.on('rename', (file, oldPath))` — fires on both rename and move. If the file has a `locus_id` frontmatter field, calls `PATCH /notes/{locus_id}/path` with the new vault-relative path.
- `vault.on('delete', (file))` — if the file has a `locus_id`, calls `PATCH /notes/{locus_id}/path` with `null` to mark the note as missing in `processing_log`. The processing record is preserved so rerun still works.

On plugin load (e.g., Obsidian opened while plugin was off), a startup reconciliation scan walks the vault for all files containing `locus_id` frontmatter and patches any `processing_log` entries whose `note_path` doesn't match. This handles bulk moves that occurred while the plugin was inactive.

### 4.2 Backend API (Python / FastAPI)

Core routes:
```
POST   /ingest/url              # body: { list_id, url } — single video, playlist, or course
POST   /ingest/freetext         # body: { list_id, query } — natural language → platform resolver
POST   /ingest/rerun/{video_id} # re-run full pipeline for an already-processed video
GET    /jobs                    # list all jobs with status
GET    /jobs/{id}               # single job status + progress
DELETE /jobs/{id}               # cancel job

POST   /lists                   # create a new list
GET    /lists                   # all lists
DELETE /lists/{id}              # delete list and its notes
GET    /lists/{id}/notes        # notes in a list
POST   /lists/{id}/chat         # multi-turn Q&A against a list's knowledge base (v1)
POST   /notes/{id}/chat         # Q&A scoped to a single note (v1)
PATCH  /notes/{locus_id}/path  # update note_path in processing_log (vault sync)

GET    /transcripts/{video_id}  # paginated transcript retrieval (?offset=0&limit=200)

GET    /models/whisper          # list available Whisper models with download status
POST   /models/whisper/download # queue a Whisper model download job

GET    /health                  # dependency health check
GET    /config                  # get config
PUT    /config                  # update config

POST   /ingest/freetext         # body: { list_id, query } — natural language → platform resolver (v3)
```

**Plugin ingestion flow:** The user creates a list first (or selects an existing one), then submits a URL or free-text query into it. Ingestion without a list is not permitted — there is no "Uncategorized" default.

**Deduplication:** Before queuing a video, the backend checks `processing_log` for an existing `video_id`. If found, the job is skipped silently. The user can override this explicitly via `POST /ingest/rerun/{video_id}`, which re-runs the full pipeline and overwrites the existing note and ChromaDB entries.

### 4.3 Job Queue & Processing Log

SQLite serves two purposes here — a transient job queue and a permanent processing log. These are kept in separate tables with different lifecycles.

**Lists are not stored in SQLite.** A list is defined by the vault: creating a list means creating `{vault}/Locus/{list_name}/_overview.md` with a `locus_list_id` in its frontmatter. The backend discovers lists by scanning vault folders and reading overview frontmatter. This keeps the vault as the single source of truth and means a user renaming a list folder in Obsidian is automatically reflected — no separate sync required.

**Job states:** `queued → downloading → transcribing → extracting → writing → done | failed`

A background worker loop picks up queued jobs sequentially (or with configurable parallelism). Completed and failed job rows can be pruned after a retention window. The worker surviving restarts is guaranteed by reading `queued` and `in_progress` rows on startup.

```sql
-- Ephemeral: pruned after completion
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

-- Permanent: one row per successfully processed video
CREATE TABLE processing_log (
    video_id            TEXT PRIMARY KEY,   -- platform-native ID (e.g. bvid)
    platform            TEXT,
    list_id             TEXT,               -- matches locus_list_id in _overview.md frontmatter
    note_path           TEXT,               -- vault-relative path to the note file
    transcript_path     TEXT,               -- absolute path to ~/.locus/transcripts/{video_id}.txt
    whisper_model       TEXT,               -- e.g. large-v3
    ai_model            TEXT,               -- model used for extraction
    vision_enabled      BOOLEAN,
    processed_at        DATETIME,
    settings_snapshot   JSON                -- full config snapshot at time of processing
);
```

The `processing_log` table is what powers deduplication ("has this video been processed?") and re-processing decisions ("was this transcribed with an older model?"). It also decouples versioning history from the Obsidian vault — note metadata stays human-readable while pipeline metadata stays in the log.

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

Auth is handled per-adapter. Bilibili uses cookie-based session stored (encrypted) in config. When a download returns a 403, the adapter raises `AuthRequiredError` and the job transitions to a `needs_auth` state, prompting the user via the plugin.

### 4.5 Processing Pipeline

Per video, once downloaded:

```
1. Extract audio (FFmpeg)
2. Transcribe audio → raw Whisper segments (Faster Whisper)
3. Write raw segments to transcript file (~/.locus/transcripts/{video_id}.txt)
4. Merge consecutive segments into RAG chunks (~200-400 tokens each)
5. LLM pass: extract title, summary, key points with timestamps
6. (v1) If vision enabled: sample frames → multimodal LLM pass
7. Write video note ({video_title}.md)
8. Embed RAG chunks into ChromaDB vector store
9. Update list overview note
10. Write processing_log entry
```

**Chunking strategy:** Whisper raw segments (~5–15s each) are merged greedily until the chunk reaches a target token count (~300 tokens). Each chunk stores `timestamp_start`, `timestamp_end`, and `sequence_index` in ChromaDB metadata. This decouples the transcript source of truth from the RAG chunk size — re-chunking or re-embedding never requires re-transcription.

### 4.6 Vector Store (ChromaDB)

Used for RAG in the Q&A engine. Each chunk is a merge of consecutive Whisper segments, stored with metadata:

```json
{
  "note_id": "...",
  "list_id": "...",
  "video_title": "...",
  "timestamp_start": 142,
  "timestamp_end": 198,
  "sequence_index": 7,
  "text": "..."
}
```

Q&A queries are scoped by `list_id` (list-level chat) or `note_id` (single-note chat).

### 4.7 Transcript Storage & Serving

Raw Whisper segments are stored outside the vault at `~/.locus/transcripts/{video_id}.txt`. This keeps them invisible to Obsidian's search, graph, and file explorer while remaining directly readable as plain text files.

The backend exposes a paginated endpoint for the plugin's transcript viewer:

```
GET /transcripts/{video_id}?offset=0&limit=200   # returns lines by segment index
```

The plugin renders transcript on demand in a read-only side panel — the user clicks "View Transcript" on a note to open it. Segments are loaded progressively on scroll.

`~/.locus/` directory layout:
```
~/.locus/
  locus.db                  # SQLite (jobs + processing_log)
  transcripts/
    {video_id}.txt          # raw Whisper segments, one per line: [HH:MM:SS] text
  chroma/                   # ChromaDB data directory
  downloads/                # temporary video files, cleaned up after processing
```

### 4.7 Free-Text Ingestion Resolver (v3)

When the user submits a natural language query like `"free AI lessons by Andrew Ng"`, the backend:

1. Sends the query to the LLM with a structured prompt to extract intent: platform hint, search terms, content type
2. Dispatches to the matching platform's search skill (one skill per platform)
3. Returns a list of candidate playlists/videos for user confirmation before ingesting
4. On confirmation, creates jobs as normal

Each platform skill is a self-contained module:
```python
class BilibiliSearchSkill:
    def search(self, query: str) -> list[SearchResult]

class YouTubeSearchSkill:
    def search(self, query: str) -> list[SearchResult]
```

The resolver always requires user confirmation before bulk ingestion — it never auto-starts jobs from free-text.

---

## 5. Data Model — Obsidian Notes

### 5.1 Vault Layout

```
{vault}/Locus/
  {list_name}/
    _overview.md
    {video_title}.md
    attachments/
      {bvid}_cover.jpg
```

Transcripts are stored outside the vault entirely (see §4.7) and served via the backend API. The vault contains only human-facing notes.

### 5.2 Video Note

Stored at: `{vault}/Locus/{list_name}/{video_title}.md`

Human-facing. Contains only what's useful to read.

```markdown
---
locus_id: bv1abc123
platform: bilibili
source_url: https://www.bilibili.com/video/BV1abc123
cover: Locus/{list_name}/attachments/bv1abc123_cover.jpg
duration: 3842
list_id: list_xyz
processed_at: 2026-03-30T10:00:00
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

### 5.2 List Overview Note

Stored at: `{vault}/Locus/{list_name}/_overview.md`

```markdown
---
locus_list_id: list_xyz
video_count: 12
last_updated: 2026-03-30T10:00:00
---

# {List Name} — Overview

## Outline
{LLM-generated outline synthesizing all videos in the list}

## Videos
- [[Video Title 1]]
- [[Video Title 2]]
...

## Supplementary Notes
{User-editable free-text area — v1}
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
    "language": "auto"    // "auto" | "zh" | "en" — configurable per-vault in UI
  },
  "vision": {
    "enabled": false,
    "frame_sample_rate": 30,
    "model": null
  },
  "obsidian": {
    "vault_path": "/path/to/vault",
    "locus_folder": "Locus"
  },
  "backend": {
    "port": 8765,
    "worker_concurrency": 1
  }
}
```

---

## 7. Deployment Health Checks

The `/health` endpoint checks all dependencies and returns a structured status. The plugin displays this as a status panel.

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

### v0 — Core Pipeline
- Global config + deployment health
- Bilibili ingestion: single video, playlist, course
- Faster Whisper transcription
- LLM-based note generation (summary + timestamped key points)
- Obsidian note writing with frontmatter
- List creation and management
- List overview note
- Job queue with status polling in plugin
- Vault sync: automatic `note_path` reconciliation on rename, delete, and plugin load

### v1 — Enhanced Knowledge
- Multimodal vision pass (opt-in per video, configurable frame sampling)
- Per-list supplementary text area (user-editable, injected into RAG context)
- Multi-turn AI Q&A (list and note scope) with transcript citation

### v2 — Generative Outputs
- TTS configuration
- Mermaid mind map generation embedded in overview notes
- Audio summary (single-voice, LLM script + TTS — scoped to list overview)

### v3 — Expanded Sources
- YouTube adapter
- Free-text ingestion resolver with per-platform search skills
- User confirmation step before bulk ingestion from free-text

---

## 9. Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backend framework | FastAPI | Async-native, easy to expose SSE for job progress |
| Job persistence | SQLite (two tables) | Zero-infra; `jobs` table is ephemeral queue, `processing_log` is permanent versioning history. Lists are vault-backed, not stored in SQLite. |
| Vector store | ChromaDB | Local, no server process needed, Python-native |
| Transcription | Faster Whisper | Best local quality/speed tradeoff, CUDA support |
| Transcript storage | `~/.locus/transcripts/` (outside vault) | Invisible to Obsidian by default; readable as plain text or via plugin panel; decoupled from RAG chunk strategy |
| RAG chunking | Greedy merge of Whisper segments (~300 tokens) | Balances embedding quality with timestamp granularity; source transcript is never modified |
| Note format | Markdown + YAML frontmatter | Obsidian-native, queryable via Dataview plugin |
| Plugin language | TypeScript | Required for Obsidian plugins |
| Auth storage | Encrypted local config file | Keeps credentials off disk in plaintext |

---

## 10. Open Questions

1. ~~**Whisper language detection**~~ — Resolved: UI dropdown with `auto / zh / en` options. Stored in config, applied globally.
2. ~~**Note deduplication**~~ — Resolved: skip duplicate `video_id` silently on ingest. User can explicitly rerun via `POST /ingest/rerun/{video_id}`.
3. ~~**List assignment**~~ — Resolved: user always creates or selects a list before ingesting. No default list. Ingestion without a list is not permitted.
4. ~~**Vault sync**~~ — Resolved: auto-reconciled in v0 via Obsidian vault events. See §4.1.
5. ~~**Chunk strategy for RAG**~~ — Resolved: greedy merge of consecutive Whisper segments to ~300 token target. Raw segments stored separately as source of truth; re-chunking does not require re-transcription.
6. ~~**List storage**~~ — Resolved: lists are vault-backed. A list is defined by `{vault}/Locus/{name}/_overview.md` with `locus_list_id` frontmatter. The backend scans vault folders to discover lists; no `lists` table in SQLite. Renaming a folder in Obsidian is automatically reflected without a separate sync step.
