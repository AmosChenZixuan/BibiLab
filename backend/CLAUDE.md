# Backend

Python/FastAPI backend. Managed with `uv`.

## Commands

```bash
uv sync --dev                        # Install all dependencies
uv run python -c "import torch; print(torch.cuda.is_available())"  # Verify CUDA
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run pytest                        # All tests
uv run pytest -m "not integration"   # Fast unit subset (skips client/real-SQLite tests)
uv run pytest tests/test_ingest.py -v  # Single test file
uv run pytest --cov=bibilab --cov-report=term-missing  # Coverage (requires pytest-cov)
uv run python -m bibilab.main       # Start server (localhost:8765)
```

## Code Layout — `src/bibilab/`

```
routers/          — one APIRouter per module; aggregated in main.py
  auth.py           /auth/bilibili/* (QR login, cookie management)
  chat.py           /lists/:id/chat (SSE streaming + cancel), /lists/:id/chat/:msg_id/stream (reattach), /lists/:id/conversation (CRUD); stream_with_tools loop; classify_error (SDK exception → i18n error code)
  lists.py          /lists/* (CRUD)
  ingest.py         /ingest/url (POST)
  sources.py        /sources/* (source content, covers, rerun, PATCH facets manual edit)
models/           — Pydantic request/response models + domain errors
  chat.py           ChatRequest, MessageResponse, ConversationResponse
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), StreamEvent/ToolCall/ToolDefinition dataclasses
  audio.py          FFmpeg audio extraction (video → .wav)
  transcribe.py     FunASR AutoModel (SenseVoice/Whisper) + CAM++ diarization → VAD segments w/ speaker labels
  chunk.py          greedy segment merger → ~300-token RAG chunks
  digest.py         LLM summary + keywords + facets (series_name, sequence_number, season_number) → denormalized into sources
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  rerank.py         lazy ONNX cross-encoder reranker (Xenova/bge-reranker-base, XLM-RoBERTa zh+en; batched inference)
  chat_tools.py     two tool definitions (find_passages, read_source) + execution dispatcher + chunk formatting + CitationRegistry + facet narrative builder; `_NO_MATCH_NOTE` is a fact-only string prepended when facet matching fails
  chat_summary.py   conversation compression (sliding window + LLM summary; summary is prose only — [N] markers not preserved)
  citation_parser.py incremental citation parser — strips [N] tokens from LLM deltas, emits citation SSE events with {index, source_id, chunk_ids}
  chat_runs.py       StreamBuffer + ChatRunRegistry; in-memory buffer decouples LLM producer from HTTP request lifetime
adapters/         — platform-specific download + resolution (base + bilibili)
db.py             — SQLite schema + query helpers
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
- **DB**: `asynccontextmanager` `get_db` wrapper; all queries use `?` placeholders, never f-string interpolation. `db.py` is strictly for SQL queries — no status mapping (extracted to `video_status.py`), no domain logic. The one deliberate write-time side effect is first-source thumbnail assignment in `_exec_write_source`, atomic with the source insert. Derive user-facing values in the router or a service function.
- **Imports**: stdlib → third-party → local, with blank lines between groups
- **Errors**: `HTTPException(status_code=N, detail=...)` for HTTP errors; `AuthRequiredError`, `DownloadError`, `PipelineError` for domain errors
- **LLM content blocks**: Never assume `msg.content[0]` is a TextBlock. Filter by `block.type == "text"` — some providers return ThinkingBlock or other types first. Use `next((b for b in msg.content if b.type == "text"), None)` with a None default to avoid StopIteration in async contexts.
- **FTS5 input**: Always pass user query strings through `_escape_fts_query()` (db.py) before MATCH evaluation. FTS5 treats bare `OR`, `*`, `:`, `^` as operators — unescaped user input raises `OperationalError`.

## Testing

- **Seed via factories, not `db.py`**: use `tests/factories.py` — `SourceFactory.build(list_id, **overrides)`, `ConversationFactory.build`, `MessageFactory.build`. Never insert through production helpers in test setup. `SourceFactory` delegates to `write_source_with_segments`, so a `sources` column change only touches `_DEFAULTS`.
- **LLM mocking via conftest fixtures**: `mock_stream_llm` (chat `stream_llm` seam) and `mock_call_llm` (digest/chat_summary/worker `_call_llm` seams). Configure with `.return_value` / `.side_effect`; don't hand-roll `patch("...stream_llm")` per test.
- **Integration marker**: any test that drives the `client` fixture or hits real SQLite/Chroma carries module-level `pytestmark = pytest.mark.integration` (place after imports). `pytest -m "not integration"` is the fast unit lane.
- **Home isolation**: the `tmp_bibilab_home` fixture sets `BIBILAB_HOME` — one env seam that `bibilab_home()` honors. Don't re-add per-module `patch("...bibilab_home")`.
- **Mock only external boundaries** (LLM, models, network). Use real SQLite/Chroma — they're embedded and spun up per-test in a temp dir; no DB fakes.

## Database Schema

### `lists`

| Column | Notes |
|---|---|
| `thumbnail_source_id` | FK to `sources.id`, nullable |

### `jobs` — ephemeral queue

| Column | Notes |
|---|---|
| `type` | `"ingest"` \| `"model_download"` \| `"artifact"` |
| `status` | `queued` → `downloading` → `transcribing` → `processing` → `done` \| `failed` \| `needs_auth` |
| `progress` | 0–100 |
| `error` | Error message, nullable |
| `meta` | JSON blob: `{ list_id, video_id, title, cover_url, platform, source_url, rerun, ... }` |

### `sources` — active video catalog

| Column | Notes |
|---|---|
| `video_id` | Platform-native ID (e.g. `bvid`) |
| `title`, `summary`, `keywords` | Denormalized from platform metadata + LLM output |
| `language`, `uploader`, `duration_seconds` | Video metadata |
| `whisper_model`, `ai_model`, `vision_enabled` | Processing config at ingest time |
| `cover_url`, `processed_at`, `settings_snapshot` | Ingest-time state |
| `series_name`, `sequence_number`, `season_number` | Facet metadata extracted by LLM digest (nullable) |

### `artifacts` — Lab-generated content

| Column | Notes |
|---|---|
| `name` | User-visible name (initial = type label, renameable) |
| `type` | `"brief"` \| `"study_guide"` \| `"blog_post"` \| `"custom_report"` |
| `prompt` | Exact prompt string submitted |
| `source_ids` | JSON array of source UUIDs |
| `status` | `generating` → `done` \| `failed` |
| `content_path` | Relative path, e.g. `artifacts/{id}.md`, nullable |
| `error` | Error message, nullable |

### `conversations` — one per list

| Column | Notes |
|---|---|
| `list_id` | FK to `lists.id`, UNIQUE, ON DELETE CASCADE |
| `summary` | Rolling LLM-generated summary, nullable |
| `active_stream_message_id` | FK to `messages.id`, nullable; points to the currently streaming assistant message |
| `updated_at` | ISO timestamp |

### `messages` — chat history

| Column | Notes |
|---|---|
| `conversation_id` | FK to `conversations.id`, ON DELETE CASCADE, indexed |
| `role` | `"user"` \| `"assistant"` \| `"tool"` |
| `content` | Message text |
| `status` | `"streaming"` → `"done"` \| `"failed"` \| `"cancelled"` |
| `error` | Error code (e.g. `llm_rate_limit_error`, `internal_error`), nullable; set on producer failure or server restart sweep; mapped by `classify_error()` from SDK exceptions; frontend resolves via `chat.errors.*` i18n keys |
| `metadata` | JSON blob, nullable. Shape: `{"tool_calls": [...], "rag": {"calls": [{"tool_name", "query" \| null, "source_id"?, "source_title"?, "source_coverage" (find_passages only), "context" (find_passages only, reconstructed at terminal `rag` event from citation registry), "candidates_evaluated", "sources_with_hits", "sources_total", "reranked", "scoped_pool_size", "facet_scope"}]}}`. `tool_name` is `"find_passages"` or `"read_source"`; `read_source` rows have `query: null`, empty `context[]`, and `source_id`/`source_title` set. `scoped_pool_size` is the **full source pool**; `facet_scope.matched_count` carries the facet-narrowed count. Set by `run_chat_turn` post-stream from tool `tool_result` events. No migration for legacy persisted messages — the ledger renders best-effort from whatever fields exist. |

### `transcript_segments` — punctuated sentence segments

| Column | Notes |
|---|---|
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

`meta`, `delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`

`meta` is the first event of every stream, carrying `{message_id}` so the client can wire cancel-by-id before the first delta (constant `SSE_EVENT_META`). The other eight are also `stream_llm` discriminators; `meta` is emitted only by the HTTP handler.

### System prompt context

- **Grounding prompt**: `build_grounding_prompt(response_language)` produces a four-section markdown document: `## Workflow` (tool routing), `## Grounding` (excerpts-only), `## Citation` (`[N]` markers), `## Style` (direct, no follow-up). `response_language` is interpolated ≥2× to match the user's UI/config language for both directives and fallback refusals. Teaches the LLM subject decomposition — one subject → one tool, multiple subjects (comparisons, multi-episode) → parallel calls, one per subject, picking the appropriate tool per subject.
- **Source scoping**: No per-turn source list in the prompt. The system prompt is static-per-language, so Anthropic prompt caching covers the whole grounding prefix. Search scope is set solely by deterministic facet matching: the LLM passes `sequence_number` / `season_number` to `find_passages` / `read_source` only when the question references them; the backend matches them against `sources`. When facet matching finds no source (fail-open to the full pool), the tool result carries `_NO_MATCH_NOTE` as a fact-only LLM-visible prefix so the model states the degraded scope before answering.

### Chat execution

```
stream_with_tools(stream_llm loop):
    LLM yields tool_call (find_passages) → execute_find_passages() → hybrid_search (BM25 + vector RRF) → rerank → top-k (no gate)
    → tool_call_start emitted → tool_result (with citation-formatted _chunks) fed back to LLM → second turn yields text with [N] citations
    → parse_delta strips [N] markers, emits citation SSE events ({index, source_id, chunk_ids}) interleaved with delta events
    LLM yields tool_call (read_source) → execute_read_source() → resolves to one source (fail-closed) → continuous transcript
    → tool_result fed back to LLM
    LLM yields delta + done directly (no tool) → exit loop
→ yield delta + tool_call_start + tool_result + citation + done events (SSE)
→ persist assistant message with rag metadata → asyncio.create_task: maybe_compress_conversation
```

- **Producer/consumer split**: `run_chat_turn` (async Task) writes SSE events into `StreamBuffer`; `_sse_consumer` reads from buffer. Decouples LLM lifetime from HTTP request.
- `stream_with_tools` wraps `stream_llm` in a bounded loop (max 3 iterations). Both tools are loopback: their results feed back for another LLM turn (allowing `find_passages → read_source` escalation). When iterations are exhausted, `active_tools` is forced to `[]` so the LLM synthesizes from accumulated results instead of yielding a hard error. If tools were used but no text was generated, a forced follow-up LLM call (no tools) ensures the user always gets an answer. `parallel_retrieve` log records hedge-vs-decomposed rate when retrieve-family fires >1 call in one iteration, for offline analysis.
- **Preamble streaming**: Text generated before a loopback tool call is streamed to the client immediately (parsed incrementally via `parse_delta`). Trade-off: short filler like "Let me look that up..." reaches the client before the retrieve runs.
- **Subject decomposition + parallel retrieval**: The grounding prompt instructs the LLM to first decompose the user's message into distinct subjects (entities, episodes, items compared). One subject → one retrieval call. Multiple subjects (`'A 和 B 的区别'`, `'第一集 xxx 第三集 yyy'`) → parallel calls, one per subject, picking the appropriate tool per subject.
- **Cross-turn reuse**: Prior-turn tool exchanges (find_passages, read_source) are dropped entirely from history replay by `expand_message_for_provider`; only synthesized prose survives into the next turn's LLM context. To answer from prior evidence the LLM must retrieve again this turn. The current turn's fresh tool calls are full-text (they go through `stream_with_tools`, not `expand_message_for_provider`).
- **Speaker-turn reconstruction**: For the retained top-k only, `execute_retrieve` runs one batched `get_segments_for_ranges` query over each chunk's `(source_id, seg_start..seg_end)` and renders the LLM-facing fence body + `context[].preview` as grouped speaker turns via the shared `format_turns` helper (`[S{N}·SPK{k} @MM:SS]`). Speakers are namespaced per source by citation index `N`. Rerank + chroma/FTS still use raw `chunk.content` (embedder parity) — reconstruction is presentation-only.
- Retrieve result `_chunks` are grouped by source under `===== Source [N]: "title" =====` fences then formatted as citation-ready `[N]: "content"` strings for the LLM; stripped from the client-bound `tool_result` SSE payload by `strip_internal()`.
- `stream_llm` supports both OpenAI and Anthropic protocols via `AsyncOpenAI`/`AsyncAnthropic`
- **Final `rag` event**: In the `finally` block, after `context[]` is reconstructed from the citation registry, `run_chat_turn` emits one `rag` SSE event carrying the authoritative persisted-shape `rag.calls` just before the terminal event. The client replaces its incrementally-built ledger so expand works post-stream without a manual reload (the streaming `tool_result` payload omits `context[]`).
- **Lifecycle**: Startup sweep marks leftover `status='streaming'` rows as `failed`. Shutdown cancels all tasks, drains with 5s timeout. Buffer eviction fires 60s after terminal status.
- Compression: triggered when message count > 30; keeps sliding window of 10; summarizes older messages via `_call_llm` in `asyncio.create_task`. The summary is prose only — the compression prompt does **not** preserve `[N]` markers (deliberate; see `docs/citation_system.md`). Only post-window messages retain live citations; the summary is injected into the system prompt on subsequent requests.

### Prompt-trace observability (opt-in)

Set `rag.debug_prompts: true` in `~/.bibilab/config.json` to dump each
LLM call. For each chat message, a directory is created at
`~/.bibilab/debug/{message_id}/` and one `call{N}.json` is written per
`stream_llm` call (one per tool-using iteration + the forced-synthesis
follow-up + any later iterations). Each file is a JSON object with
`{system, tools, messages, response: {text, tool_calls}}` — the full
LLM call (input AND response), with no reformatting. **Off by default**
(no I/O, no behavior change). Writes are best-effort: failures are
logged and never break the turn.

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
  "rag": { "max_distance": 0.8, "hybrid_enabled": true, "reranking_enabled": true, "debug_prompts": false }
}
```
Reranker model is fixed to `Xenova/bge-reranker-base` (XLM-RoBERTa, Chinese + English). `FIND_PASSAGES_TOP_K = 8`. Opt-in prompt-trace dump writes one `call{N}.json` per `stream_llm` call under `~/.bibilab/debug/{message_id}/` when `rag.debug_prompts` is true; off by default.
