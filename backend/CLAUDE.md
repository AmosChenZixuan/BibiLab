# Backend

Python/FastAPI backend. Managed with `uv`.

## Commands

```bash
uv sync --dev                        # Install all dependencies
uv sync --extra cuda                 # Install optional CUDA libs (nvidia-cublas-cu12)
uv run python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"  # Verify CUDA
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run pytest                        # All tests
uv run pytest tests/test_ingest.py -v  # Single test file
uv run pytest --cov=bibilab --cov-report=term-missing  # Coverage (requires pytest-cov)
uv run python -m bibilab.main       # Start server (localhost:8765)
```

## Code Layout — `src/bibilab/`

```
routers/          — one APIRouter per module; aggregated in main.py
  auth.py           /auth/bilibili/* (QR login, cookie management)
  chat.py           /lists/:id/chat (SSE streaming), /lists/:id/conversation (CRUD)
models/           — Pydantic request/response models + domain errors
  chat.py           ChatRequest, MessageResponse, ConversationResponse
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), tool/stream dataclasses
  chat_tools.py     tool definitions (generate_report) + execution dispatcher
  chat_summary.py   conversation compression (sliding window + LLM summary)
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  rerank.py         lazy cross-encoder reranker (sentence-transformers, singleton)
  route.py          LLM-based query classifier (factual/breadth/analytical → focused/broad)
adapters/         — platform-specific download + resolution (base + bilibili)
db.py             — SQLite schema + query helpers
config.py         — settings persisted to ~/.bibilab/config.json
worker.py         — SQLite-polling job dispatcher; accepts config/adapter/home via constructor for testability
cleanup.py        — resource cleanup utilities
whisper_models.py — Whisper model management
```

## Conventions

- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for classes and Pydantic models
- **Pydantic models**: `{Operation}Request` / `{Operation}Response` suffix; enums use `PascalCase` name, `UPPERCASE` values
- **Router pattern**: one `APIRouter` per file, no prefix/tags; routes carry full paths
- **DB**: `asynccontextmanager` `get_db` wrapper; all queries use `?` placeholders, never f-string interpolation. `db.py` is strictly for SQL queries — no status mapping, no side effects (e.g., auto-setting thumbnails), no domain logic. If you need to derive a user-facing value from raw data, do it in the router or a service function.
- **Imports**: stdlib → third-party → local, with blank lines between groups
- **Errors**: `HTTPException(status_code=N, detail=...)` for HTTP errors; `AuthRequiredError`, `DownloadError`, `PipelineError` for domain errors
- **LLM content blocks**: Never assume `msg.content[0]` is a TextBlock. Filter by `block.type == "text"` — some providers return ThinkingBlock or other types first. Use `next((b for b in msg.content if b.type == "text"), None)` with a None default to avoid StopIteration in async contexts.
- **FTS5 input**: Always pass user query strings through `_escape_fts_query()` (db.py) before MATCH evaluation. FTS5 treats bare `OR`, `*`, `:`, `^` as operators — unescaped user input raises `OperationalError`.

## Database Schema

### `lists`

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `name` | User-visible name |
| `thumbnail_source_id` | FK to `sources.id`, nullable |
| `created_at` | ISO timestamp |

### `jobs` — ephemeral queue

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `type` | `"ingest"` \| `"model_download"` \| `"artifact"` |
| `status` | `queued` → `downloading` → `transcribing` → `processing` → `done` \| `failed` \| `needs_auth` |
| `progress` | 0–100 |
| `error` | Error message, nullable |
| `meta` | JSON blob: `{ list_id, video_id, title, cover_url, platform, source_url, rerun, ... }` |

### `sources` — active video catalog

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `video_id` | Platform-native ID (e.g. `bvid`) |
| `platform` | Platform name |
| `list_id` | FK to `lists.id` |
| `title`, `summary`, `keywords` | Denormalized from platform metadata + LLM output |
| `language`, `uploader`, `duration_seconds` | Video metadata |
| `transcript_path` | Relative path from `~/.bibilab/`, nullable |
| `whisper_model`, `ai_model`, `vision_enabled` | Processing config at ingest time |
| `cover_url`, `processed_at`, `settings_snapshot` | Ingest-time state |

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
| `content_path` | Relative path, e.g. `artifacts/{id}.md`, nullable |
| `error` | Error message, nullable |
| `created_at` | ISO timestamp |

### `conversations` — one per list

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `list_id` | FK to `lists.id`, UNIQUE, ON DELETE CASCADE |
| `summary` | Rolling LLM-generated summary, nullable |
| `mode` | `"auto"` \| `"focused"` \| `"broad"`, default `'auto'`. Read by chat router to decide retrieval mode and whether to run the query classifier. |
| `created_at` | ISO timestamp |
| `updated_at` | ISO timestamp |

### `messages` — chat history

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `conversation_id` | FK to `conversations.id`, ON DELETE CASCADE, indexed |
| `role` | `"user"` \| `"assistant"` \| `"tool"` |
| `content` | Message text |
| `metadata` | JSON blob, nullable. Shape: `{"tool_calls": [...], "rag": {mode, candidates_evaluated, sources_with_hits, sources_total, sources}}`. The wire SSE envelope (`{"type": "rag_meta", ...}`) is unwrapped before storage. |
| `created_at` | ISO timestamp |

### `chunks_fts` — FTS5 virtual table

BM25-ranked full-text index over transcript chunks. Populated by `populate_fts()` during the embed stage; cleared per-video on re-ingest. Columns: `content`, `video_id`, `video_title`, `timestamp_start`, `timestamp_end`, `chunk_id`. Query via `query_fts_rows()` in `db.py`.

### `query_classifications` — routing log (training data)

Append-only log of LLM classifier outputs. Columns: `id`, `list_id`, `query_text`, `query_type`, `effective_mode`, `created_at`. Insert wrapped in try/except — failures are logged but do not break the chat request.

## Ingestion Pipeline

```
POST /ingest/url → resolve → dedup check → create job(s)
  → worker: download → audio → transcribe → chunk → digest ∥ embed → write_source
```

- Dedup via `sources.video_id`; skip if found unless `?rerun=true`
- Delete removes transcript file, ChromaDB embeddings, and `sources` row

### Pipeline stages (per video)

1. **download** → temp video file
2. **audio** → strip to .wav via FFmpeg
3. **transcribe** → Faster Whisper → raw segments with timestamps
4. **chunk** → greedily merge consecutive Whisper segments (~5–15s each) until ~300 tokens. Each chunk stores `timestamp_start`, `timestamp_end`, `sequence_index` in ChromaDB metadata.
5. **digest** → LLM: summary, keywords → denormalized into `sources` (parallel with embed)
6. **embed** → store chunks in ChromaDB with per-video and per-list scope (parallel with digest)
7. **write_source** → upsert row into `sources`

## Chat Pipeline

```
POST /lists/:id/chat (SSE)
  → get_or_create_conversation → load history + stored mode
  → if mode == "auto" and query_routing_enabled: classify_query → effective_mode
  → retrieve(): hybrid_search (BM25 + vector RRF, pool 30) → rerank → aggregate (broad, no top-k cap) or top-k (focused) → floor filter
  → sources_with_hits reflects candidate-pool coverage; result_chunks reflects actual LLM input
  → stream_llm (system prompt + RAG context + history)
  → yield rag_meta + delta/tool_result/done events
  → persist assistant message → BackgroundTask: maybe_compress_conversation
```

- `stream_llm` supports both OpenAI and Anthropic protocols via `AsyncOpenAI`/`AsyncAnthropic`
- Tool calling: LLM can invoke `generate_report` → creates artifact job
- Compression: triggered when message count > 30; keeps sliding window of 20; summarizes older messages via `_call_llm` in `asyncio.to_thread`
- Summary injected into system prompt on subsequent requests

## Platform Adapters

```python
class PlatformAdapter:
    def resolve_flat(url) -> PlaylistMeta
    async def get_videos_metadata(video_ids) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]
    def download(video_id, source_url) -> Path
```

v0: `BilibiliAdapter` — single video. Cookie-based auth in config.
403 → `AuthRequiredError` → job `needs_auth` → UI prompts user.

**QR login flow**: `POST /auth/bilibili/qr` → get `{url, key}` → UI polls `GET /auth/bilibili/qr/status?key=...` (query param, not path param — avoids key in server logs) → on success, cookie saved to config.

**Cookie file**: `_cookie_file()` converts the raw cookie string to Netscape HTTP Cookie File format (yt-dlp requirement). A module-level `_cookie_file_cache` skips the disk write when the cookie string is unchanged.

## Configuration Schema

```json
{
  "accounts": { "bilibili": { "cookie": "", "last_verified": "" } },
  "ai": { "protocol": "openai|anthropic", "model": "", "api_key": "", "base_url": null, "output_language": "ui" },
  "transcription": { "engine": "faster-whisper", "model_size": "large-v3", "device": "cuda|cpu", "language": "auto" },
  "vision": { "enabled": false, "frame_sample_rate": 30, "model": null },
  "backend": { "port": 8765, "worker_concurrency": 1 },
  "rag": { "max_distance": 0.8, "hybrid_enabled": true, "reranking_enabled": true, "query_routing_enabled": true, "rerank_min_score": 0.0 }
}
```
