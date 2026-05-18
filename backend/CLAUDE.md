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
  chat.py           /lists/:id/chat (SSE streaming + cancel), /lists/:id/chat/:msg_id/stream (reattach), /lists/:id/conversation (CRUD); stream_with_tools loop; classify_error (SDK exception → i18n error code)
  lists.py          /lists/* (CRUD), /lists/:id/overview (POST)
  ingest.py         /ingest/url (POST)
  sources.py        /sources/* (source content, covers, rerun)
models/           — Pydantic request/response models + domain errors
  chat.py           ChatRequest, MessageResponse, ConversationResponse
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), StreamEvent/ToolCall/ToolDefinition dataclasses
  audio.py          FFmpeg audio extraction (video → .wav)
  transcribe.py     Faster Whisper transcription → raw segments
  chunk.py          greedy segment merger → ~300-token RAG chunks
  digest.py         LLM summary + keywords + facets (series_name, sequence_number, sequence_kind, season_number) → denormalized into sources
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  extract.py        LLM knowledge synthesis (overview generation)
  rerank.py         lazy ONNX cross-encoder reranker (Xenova/bge-reranker-base, XLM-RoBERTa zh+en; batched inference)
  chat_tools.py     tool definitions (retrieve + query_list_metadata + generate_report) + execution dispatcher + chunk formatting + CitationRegistry + trivial-acknowledgment guard (trivial_ack_note)
  chat_summary.py   conversation compression (sliding window + LLM summary; preserves [N] RAG citations)
  citation_parser.py incremental citation parser — strips [N] tokens from LLM deltas, emits citation SSE events with {index, source_id, chunk_ids}
  chat_runs.py       StreamBuffer + ChatRunRegistry; in-memory buffer decouples LLM producer from HTTP request lifetime
adapters/         — platform-specific download + resolution (base + bilibili)
db.py             — SQLite schema + query helpers (840 lines)
video_status.py   — derive_video_statuses (status mapping extracted from db.py per Code Health Rule #4)
config.py         — settings persisted to ~/.bibilab/config.json; includes models_dir() helper
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
| `series_name`, `sequence_number`, `sequence_kind`, `season_number` | Facet metadata extracted by LLM digest (nullable) |

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
| `active_stream_message_id` | FK to `messages.id`, nullable; points to the currently streaming assistant message |
| `created_at` | ISO timestamp |
| `updated_at` | ISO timestamp |

### `messages` — chat history

| Column | Notes |
|---|---|
| `id` | UUID, primary key |
| `conversation_id` | FK to `conversations.id`, ON DELETE CASCADE, indexed |
| `role` | `"user"` \| `"assistant"` \| `"tool"` |
| `content` | Message text |
| `status` | `"streaming"` → `"done"` \| `"failed"` \| `"cancelled"` |
| `error` | Error code (e.g. `llm_rate_limit_error`, `internal_error`), nullable; set on producer failure or server restart sweep; mapped by `classify_error()` from SDK exceptions; frontend resolves via `chat.errors.*` i18n keys |
| `metadata` | JSON blob, nullable. Shape: `{"tool_calls": [...], "rag": {"calls": [{"query", "expected_hits", "candidates_evaluated", "sources_with_hits", "sources_total", "source_coverage", "context": [{"chunk_id", "citation_index", "source_id", "source_title", "timestamp_start", "timestamp_end", "rerank_score", "preview"}], "dropped_by_gate", "reranked", "scope_choice", "excluded_count", "scoped_pool_size", "gate_margin"}]}}`. Set by `run_chat_turn` post-stream from retrieve `tool_result` events (one entry per retrieve call). No backward compatibility with pre-I-4 persisted messages — clear DB if needed after this issue ships. |
| `created_at` | ISO timestamp |

### `chunks_fts` — FTS5 virtual table

BM25-ranked full-text index over transcript chunks. Populated by `populate_fts()` during the embed stage; cleared per-video on re-ingest. Columns: `content`, `video_id`, `video_title`, `timestamp_start`, `timestamp_end`, `chunk_id`. Query via `query_fts_rows()` in `db.py`.

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
POST /lists/:id/chat (SSE) — creates user + assistant rows atomically, spawns async producer
  → asyncio.Task: run_chat_turn writes events into StreamBuffer
  → POST handler returns SSE stream consuming from buffer (late-subscriber-safe replay)
  → producer persists final content + status + metadata in finally block
  → producer fires fire-and-forget: evict buffer after grace, compress if done

GET  /lists/:id/chat/{msg_id}/stream — reattach to an active stream (204 if evicted)
POST /lists/:id/chat/{msg_id}/cancel — cancel producer task, persists status='cancelled'

Startup: sweep_orphaned_streams marks leftover streaming rows as failed
Shutdown: cancel all active tasks, drain with 5s timeout
```

### SSE event types

`delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`

### System prompt context

- **Grounding prompt**: `build_grounding_prompt(response_language)` produces a four-section markdown document: `## Workflow` (tool routing), `## Grounding` (excerpts-only), `## Citation` (`[N]` markers), `## Style` (direct, no follow-up). `response_language` is interpolated ≥2× to match the user's UI/config language for both directives and fallback refusals.
- **Source list**: Before streaming, the system prompt includes a numbered list of all sources: `[N] Title (keywords)`. The LLM scans for clearly unrelated sources and passes their numbers to `retrieve(exclude_source_ids=[...])`, scoping search to everything else. Anthropic prompt caching makes these tokens near-free after the first turn.

### Chat execution

```
trivial_ack_note(current_msg) → strip prior tool_blocks + inject no-retrieve note on a trivial ack; otherwise pass through (prior excerpts replay; LLM self-judges reuse from the grounding prompt)
stream_with_tools(stream_llm loop):
    LLM yields tool_call (retrieve) → execute_retrieve() → hybrid_search (BM25 + vector RRF) → rerank → _diverse_top_k
    → tool_call_start emitted → tool_result (with citation-formatted _chunks) fed back to LLM → second turn yields text with [N] citations
    → parse_delta strips [N] markers, emits citation SSE events ({index, source_id, chunk_ids}) interleaved with delta events
    LLM yields tool_call (query_list_metadata) → execute_query_list_metadata() → tool_result → fed back to LLM
    LLM yields tool_call (generate_report) → execute_generate_report() → tool_result → exit loop
    LLM yields delta + done directly (no tool) → exit loop
→ yield delta + tool_call_start + tool_result + citation + done events (SSE)
→ persist assistant message with rag metadata → asyncio.create_task: maybe_compress_conversation
```

- **Producer/consumer split**: `run_chat_turn` (async Task) writes SSE events into `StreamBuffer`; `_sse_consumer` reads from buffer. Decouples LLM lifetime from HTTP request.
- `stream_with_tools` wraps `stream_llm` in a bounded loop (max 3 iterations). Loopback tools (`retrieve`, `query_list_metadata`) feed results back for another LLM turn; terminal tools (`generate_report`) exit the loop. When iterations are exhausted, `active_tools` is forced to `[]` so the LLM synthesizes from accumulated results instead of yielding a hard error. If tools were used but no text was generated, a forced follow-up LLM call (no tools) ensures the user always gets an answer.
- **Preamble streaming**: Text generated before a loopback tool call is streamed to the client immediately (parsed incrementally via `parse_delta`). Trade-off: short filler like "Let me look that up..." reaches the client before the retrieve runs.
- **Sequential-retrieve guard**: After the first `retrieve` call succeeds, `retrieve` is removed from the active tool list for the remainder of the turn. A second retrieve within the same turn is rejected.
- **Cross-turn reuse**: Prior `retrieve` results replay into the LLM messages via `expand_message_for_provider`; the LLM self-judges reuse from the `build_grounding_prompt` `## Workflow` instruction (no cosine gate). `trivial_ack_note()` is the only code path that strips replayed blocks — for len<3 / pure-digit / stopword acknowledgments — and injects a no-retrieve note (unconditional: fires even when no prior retrieve exists). Reuse horizon ≈ sliding window of 20 / ~10 turns.
- Retrieve result `_chunks` are formatted as citation-ready `[N]: "content"` strings for the LLM; stripped from the client-bound `tool_result` SSE payload by `_client_tool_result()`.
- `citation_parser.parse_delta()` strips `[N]` markers from LLM output and emits `citation` SSE events with `{index, source_id, chunk_ids}`. A partial `[` at delta end is held in a buffer for the next delta. `flush_buffer()` drains remaining buffer at stream end.
- `stream_llm` supports both OpenAI and Anthropic protocols via `AsyncOpenAI`/`AsyncAnthropic`
- **Final `rag` event**: In the `finally` block, after `context[]` is reconstructed from the citation registry, `run_chat_turn` emits one `rag` SSE event carrying the authoritative persisted-shape `rag.calls` just before the terminal event. The client replaces its incrementally-built ledger so expand works post-stream without a manual reload (the streaming `tool_result` payload omits `context[]`).
- **Reattach**: Frontend reads `active_stream_message_id` from conversation; if set, GETs the stream endpoint to replay buffered events + tail live ones. Returns 204 after buffer eviction.
- **Cancel**: POST sets `status='cancelled'` + emits `cancelled` SSE event. Producer catches `CancelledError` and persists partial content.
- **Lifecycle**: Startup sweep marks leftover `status='streaming'` rows as `failed`. Shutdown cancels all tasks, drains with 5s timeout. Buffer eviction fires 60s after terminal status.
- Compression: triggered when message count > 30; keeps sliding window of 20; summarizes older messages via `_call_llm` in `asyncio.create_task`; LLM prompt instructs preservation of `[N]` RAG citations
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
  "rag": { "max_distance": 0.8, "hybrid_enabled": true, "reranking_enabled": true, "rerank_min_score": null }
}
```
Reranker model is fixed to `Xenova/bge-reranker-base` (XLM-RoBERTa, Chinese + English). `_quantile_gate` replaces the deprecated `rerank_min_score` floor — margin (bge logit units) is selected per retrieval from `_RELEVANCE_MARGIN_BY_HITS[expected_hits]` (one=1.0, few=2.0, many=2.5). `RetrievalResult.gate_margin` telemetry records the margin used. See `docs/internal/rag_tuning.md`.
