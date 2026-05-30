# Backend

Python/FastAPI backend. Managed with `uv`.

## Commands

```bash
uv sync --dev                        # Install all dependencies
uv run python -c "import torch; print(torch.cuda.is_available())"  # Verify CUDA
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
  sources.py        /sources/* (source content, covers, rerun, PATCH facets manual edit)
models/           — Pydantic request/response models + domain errors
  chat.py           ChatRequest, MessageResponse, ConversationResponse
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), StreamEvent/ToolCall/ToolDefinition dataclasses
  audio.py          FFmpeg audio extraction (video → .wav)
  transcribe.py     Faster Whisper transcription → raw segments
  chunk.py          greedy segment merger → ~300-token RAG chunks
  digest.py         LLM summary + keywords + facets (series_name, sequence_number, season_number) → denormalized into sources
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  extract.py        LLM knowledge synthesis (overview generation)
  rerank.py         lazy ONNX cross-encoder reranker (Xenova/bge-reranker-base, XLM-RoBERTa zh+en; batched inference)
  chat_tools.py     tool definitions (retrieve, survey, retrieve_scoped + query_list_metadata + generate_report) + execution dispatcher + chunk formatting + CitationRegistry + stale tool_block compaction (_summarize_stale_retrieve_block)
  chat_summary.py   conversation compression (sliding window + LLM summary; summary is prose only — [N] markers not preserved)
  citation_parser.py incremental citation parser — strips [N] tokens from LLM deltas, emits citation SSE events with {index, source_id, chunk_ids}
  chat_runs.py       StreamBuffer + ChatRunRegistry; in-memory buffer decouples LLM producer from HTTP request lifetime
adapters/         — platform-specific download + resolution (base + bilibili)
db.py             — SQLite schema + query helpers (1094 lines)
video_status.py   — derive_video_statuses (status mapping extracted from db.py per Code Health Rule #4)
config.py         — settings persisted to ~/.bibilab/config.json; includes models_dir() helper
worker.py         — SQLite-polling job dispatcher; accepts config/adapter/home via constructor for testability
cleanup.py        — resource cleanup utilities
asr_models.py     — Unified ASR model registry (Whisper + SenseVoice + diarization)
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
| `whisper_model`, `ai_model`, `vision_enabled` | Processing config at ingest time |
| `cover_url`, `processed_at`, `settings_snapshot` | Ingest-time state |
| `series_name`, `sequence_number`, `season_number` | Facet metadata extracted by LLM digest (nullable) |

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
| `metadata` | JSON blob, nullable. Shape: `{"tool_calls": [...], "rag": {"calls": [{"query", "mode", "tool_name", "candidates_evaluated", "sources_with_hits", "sources_total", "source_coverage", "context": [{"chunk_id", "citation_index", "source_id", "source_title", "timestamp_start", "timestamp_end", "rerank_score", "preview"}], "dropped_by_gate", "reranked", "scoped_pool_size", "facet_scope": {"sequence_number", "season_number", "matched_count", "no_match"}, "gate_margin"}]}}`. `scoped_pool_size` is the **full source pool** (exclude/whitelist removed); `facet_scope.matched_count` carries the facet-narrowed count. Set by `run_chat_turn` post-stream from retrieve `tool_result` events (one entry per retrieve call). No migration for legacy persisted messages — the ledger renders best-effort from whatever fields exist. |
| `created_at` | ISO timestamp |

### `transcript_segments` — punctuated sentence segments

| Column | Notes |
|---|---|
| `id` | INTEGER, primary key |
| `source_id` | FK to `sources.id`, ON DELETE CASCADE |
| `seq` | Segment order within a source |
| `start_s` | Start time in seconds |
| `end_s` | End time in seconds |
| `speaker` | Speaker label (e.g. `SPK_0`), nullable |
| `text` | Punctuated sentence text |

Index: `idx_segments_source` on `(source_id, seq)`. Replaces the on-disk `transcripts/{source_id}.txt` (P2).

### `chunks_fts` — FTS5 virtual table

BM25-ranked full-text index over transcript chunks. Populated by `populate_fts()` during the embed stage; cleared per-source on re-ingest. Columns: `content`, `source_id`, `video_title`, `timestamp_start`, `timestamp_end`, `chunk_id`, `seg_start`, `seg_end`. Query via `query_fts_rows()` in `db.py`.

## Ingestion Pipeline

```
POST /ingest/url → resolve → dedup check → create job(s)
  → worker: download → audio → transcribe → punctuate → chunk → digest ∥ embed → write_source + write_transcript_segments
```

- Dedup via `get_video_statuses` (sources + jobs); skip if processed or in-flight.
- Full re-process: DELETE /sources/:id then re-ingest.
- POST /sources/:id/rerun re-runs digest only (LLM summary, keywords, facets); transcript is reused.
- Delete removes ChromaDB embeddings and `sources` row (transcript segments cascade via FK)

### Pipeline stages (per video)

1. **download** → temp video file
2. **audio** → strip to .wav via FFmpeg
3. **transcribe** → FunASR AutoModel (SenseVoice or Whisper via WhisperWarp) → raw VAD segments with timestamps + speaker labels
4. **punctuate** → ct-punc (zh-gated) → punctuated sentence segments persisted to `transcript_segments`
5. **chunk** → greedily merge consecutive **sentence** segments to token target, split at trustworthy sentence boundary. Records `seg_start`/`seg_end` per chunk (input-index range).
6. **digest** → LLM: summary, keywords → denormalized into `sources` (parallel with embed)
7. **embed** → store chunks in ChromaDB with per-source and per-list scope (parallel with digest). Chroma metadata keys on `source_id` (+ `seg_start`/`seg_end`), not `video_id`.
8. **write_source** → upsert row into `sources`

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
- **Source scoping**: No per-turn source list in the prompt. The system prompt is static-per-language, so Anthropic prompt caching covers the whole grounding prefix. Search scope is set solely by deterministic facet matching: the LLM passes `sequence_number` / `season_number` to `retrieve_scoped` only when the question references them; the backend matches them against `sources`. Non-scoped tools (`retrieve`, `survey`) that receive facets trigger a defensive strip with warning. No `exclude_source_ids` / `source_ids` index decision. When facet matching finds no source (fail-open to the full pool), `execute_retrieve` prepends `_NO_MATCH_NOTE` to the LLM-visible `_chunks` so the model states the degraded scope before answering.

### Chat execution

```
stream_with_tools(stream_llm loop):
    LLM yields tool_call (retrieve/survey/retrieve_scoped) → execute_retrieve() → hybrid_search (BM25 + vector RRF) → rerank → _diverse_top_k
    → tool_call_start emitted → tool_result (with citation-formatted _chunks) fed back to LLM → second turn yields text with [N] citations
    → parse_delta strips [N] markers, emits citation SSE events ({index, source_id, chunk_ids}) interleaved with delta events
    LLM yields tool_call (query_list_metadata) → execute_query_list_metadata() → tool_result → fed back to LLM
    LLM yields tool_call (generate_report) → execute_generate_report() → tool_result → exit loop
    LLM yields delta + done directly (no tool) → exit loop
→ yield delta + tool_call_start + tool_result + citation + done events (SSE)
→ persist assistant message with rag metadata → asyncio.create_task: maybe_compress_conversation
```

- **Producer/consumer split**: `run_chat_turn` (async Task) writes SSE events into `StreamBuffer`; `_sse_consumer` reads from buffer. Decouples LLM lifetime from HTTP request.
- `stream_with_tools` wraps `stream_llm` in a bounded loop (max 3 iterations). Loopback tools (`retrieve`, `survey`, `retrieve_scoped`, `query_list_metadata`) feed results back for another LLM turn; terminal tools (`generate_report`) exit the loop. When iterations are exhausted, `active_tools` is forced to `[]` so the LLM synthesizes from accumulated results instead of yielding a hard error. If tools were used but no text was generated, a forced follow-up LLM call (no tools) ensures the user always gets an answer.
- **Preamble streaming**: Text generated before a loopback tool call is streamed to the client immediately (parsed incrementally via `parse_delta`). Trade-off: short filler like "Let me look that up..." reaches the client before the retrieve runs.
- **Sequential-retrieve guard**: After the first retrieve-family call succeeds (`retrieve`, `survey`, or `retrieve_scoped`), all three tools are removed from the active tool list for the remainder of the turn. A second retrieve within the same turn is rejected regardless of which variant was used first. Note: the guard is **per-iteration**, not per-call — the LLM may emit multiple retrieve-family calls in parallel within iteration 1 (legitimate for multi-subject questions); only iterations 2+ are blocked.
- **Subject decomposition + parallel retrieval**: The grounding prompt instructs the LLM to first decompose the user's message into distinct subjects (entities, episodes, items compared). One subject → one retrieval call (no hedging across variants). Multiple subjects (`'A 和 B 的区别'`, `'第一集 xxx 第三集 yyy'`) → parallel calls, one per subject, picking the appropriate tool per subject. Hedge detection: `stream_with_tools` logs `parallel_retrieve count=N names=[...] queries=[...]` when retrieve-family fires >1 call in one iteration, for offline rate analysis.
- **Cross-turn reuse**: Prior-turn retrieve results are replayed as a compact one-line tag (`_summarize_stale_retrieve_block`) — query + source titles, no excerpt text. The LLM cannot cite a prior turn's excerpts; to answer from that content it must retrieve again this turn. The current turn's fresh retrieve is full-text (it goes through `stream_with_tools`, not `expand_message_for_provider`). `query_list_metadata` / `generate_report` blocks replay in full. Reuse horizon for awareness ≈ sliding window of 10 / ~5 turns; for content, one turn.
- **Speaker-turn reconstruction**: For the retained top-k only, `execute_retrieve` runs one batched `get_segments_for_ranges` query over each chunk's `(source_id, seg_start..seg_end)` and renders the LLM-facing fence body + `context[].preview` as grouped speaker turns via the shared `format_turns` helper (`[S{N}·SPK{k} @{t}s]`). Speakers are namespaced per source by citation index `N`. Rerank + chroma/FTS still use raw `chunk.content` (embedder parity) — reconstruction is presentation-only.
- Retrieve result `_chunks` are grouped by source under `===== Source [N]: "title" =====` fences then formatted as citation-ready `[N]: "content"` strings for the LLM (#297); stripped from the client-bound `tool_result` SSE payload by `_client_tool_result()`.
- `citation_parser.parse_delta()` strips `[N]` markers from LLM output and emits `citation` SSE events with `{index, source_id, chunk_ids}`. A partial `[` at delta end is held in a buffer for the next delta. `flush_buffer()` drains remaining buffer at stream end.
- `stream_llm` supports both OpenAI and Anthropic protocols via `AsyncOpenAI`/`AsyncAnthropic`
- **Final `rag` event**: In the `finally` block, after `context[]` is reconstructed from the citation registry, `run_chat_turn` emits one `rag` SSE event carrying the authoritative persisted-shape `rag.calls` just before the terminal event. The client replaces its incrementally-built ledger so expand works post-stream without a manual reload (the streaming `tool_result` payload omits `context[]`).
- **Reattach**: Frontend reads `active_stream_message_id` from conversation; if set, GETs the stream endpoint to replay buffered events + tail live ones. Returns 204 after buffer eviction.
- **Cancel**: POST sets `status='cancelled'` + emits `cancelled` SSE event. Producer catches `CancelledError` and persists partial content.
- **Lifecycle**: Startup sweep marks leftover `status='streaming'` rows as `failed`. Shutdown cancels all tasks, drains with 5s timeout. Buffer eviction fires 60s after terminal status.
- Compression: triggered when message count > 30; keeps sliding window of 10; summarizes older messages via `_call_llm` in `asyncio.create_task`. The summary is prose only — the compression prompt does **not** preserve `[N]` markers (deliberate; see `docs/citation_system.md`). Only post-window messages retain live citations.
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
  "transcription": { "model": "sensevoice-small|large-v3", "device": "cuda|cpu", "language": "auto" },
  "vision": { "enabled": false, "frame_sample_rate": 30, "model": null },
  "backend": { "port": 8765, "worker_concurrency": 1 },
  "rag": { "max_distance": 0.8, "hybrid_enabled": true, "reranking_enabled": true }
}
```
Reranker model is fixed to `Xenova/bge-reranker-base` (XLM-RoBERTa, Chinese + English). Relevance gating uses `_quantile_gate`; margin (bge logit units) is selected per retrieval from `_RELEVANCE_MARGIN_BY_MODE[mode]` (narrow=2.0, survey=2.5). `RetrievalResult.gate_margin` telemetry records the margin used. See `docs/internal/rag_tuning.md`.
