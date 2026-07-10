# Backend

Python/FastAPI backend. Managed with `uv`.

## Commands

```bash
uv sync --dev                        # Install all deps (cpu torch by default; NVIDIA GPU: uv sync --no-default-groups --group dev --group cuda)
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
  chat.py           /lists/:id/chat (SSE streaming + cancel), /lists/:id/chat/:msg_id/stream (reattach), /lists/:id/conversation (CRUD), /debug/messages/:msg_id (prompt-trace dump read, debug_router); stream_with_tools loop; classify_error (SDK exception → i18n error code)
  eval.py           /eval/run_chat (stateless one-shot JSON chat for eval frameworks; no persistence — see "Eval endpoint" below), /eval/llm (bare _call_llm passthrough with the same llm-override merge, so the eval package needs no LLM SDK)
  lists.py          /lists/* (CRUD)
  ingest.py         /ingest/url (POST)
  sources.py        /sources/* (source content, covers, sections list, rerun, PATCH facets manual edit)
models/           — Pydantic request/response models + domain errors
  chat.py           ChatRequest, MessageResponse, ConversationResponse
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), StreamEvent/ToolCall/ToolDefinition dataclasses
  audio.py          FFmpeg audio extraction (video → .wav)
  transcribe.py     FunASR AutoModel (SenseVoice/Whisper) + CAM++ diarization → VAD segments w/ speaker labels
  chunk.py          per-section greedy segment merger → token target (`zh=800`, `en=300`); physical chunk `[seg_start, seg_end]` is fully contained in one section's range
  section.py        Section dataclass + derive_sections (token+pause boundary, target=12000) + chunk_by_sections (per-section chunking with source-global re-stamp)
  digest.py         LLM facets (series_name, sequence_number, season_number) → source facets; per-section summary + keywords → sections table (sections are the sole digest store; source carries facets only)
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  rerank.py         lazy ONNX cross-encoder reranker (Xenova/bge-reranker-base int8, XLM-RoBERTa zh+en; batched inference; single spec RERANKER_SPEC_ID; providers via interpreting_providers() so CoreML is excluded — shared with embed)
  chat_tools.py     two tool definitions (find_passages, read_section) + execution dispatcher + section fencing + CitationRegistry + facet narrative builder; `_NO_MATCH_NOTE` is a fact-only string prepended when facet matching fails
  chat_summary.py   conversation compression (sliding window + LLM summary; summary is prose only — [N] markers not preserved)
  citation_parser.py incremental citation parser — strips [N] tokens from LLM deltas, emits citation SSE events with {index, section_id, source_id, timestamp_start, chunk_ids}
  chat_runs.py       StreamBuffer + ChatRunRegistry; in-memory buffer decouples LLM producer from HTTP request lifetime
adapters/         — platform download + resolution; __init__ holds the registry (URL/platform dispatch) + CDN_DOMAINS; _ytdlp_common shared plumbing; bilibili / youtube / tiktok implementations
db.py             — SQLite schema + query helpers
video_status.py   — derive_video_statuses (status mapping extracted from db.py per Code Health Rule #4)
config.py         — settings persisted to ~/.bibilab/config.json; includes models_dir() helper
worker.py         — SQLite-polling job dispatcher; accepts config/adapter/home via constructor for testability
cleanup.py        — resource cleanup utilities
model_registry.py — Unified model dependency registry; all non-LLM downloads (ASR, VAD, diarization, embedding, reranker, punctuation) via ensure() with per-model locks + atomic .partial→rename. Holds RERANKER_SPEC_ID.
```

## Conventions

- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for classes and Pydantic models
- **Pydantic models**: `{Operation}Request` / `{Operation}Response` suffix; enums use `PascalCase` name, `UPPERCASE` values
- **Router pattern**: one `APIRouter` per file, no prefix/tags; routes carry full paths
- **DB**: `asynccontextmanager` `get_db` wrapper; all queries use `?` placeholders, never f-string interpolation. `db.py` is strictly for SQL queries — no status mapping (extracted to `video_status.py`), no domain logic. Exception: `_exec_write_source` maintains the "a non-empty list has a thumbnail" invariant by assigning the first source as cover, atomic with the insert. Derive user-facing values in the router or a service function.
- **Imports**: stdlib → third-party → local, with blank lines between groups
- **Errors**: `HTTPException(status_code=N, detail=...)` for HTTP errors; `AuthRequiredError`, `DownloadError`, `PipelineError` for domain errors
- **LLM content blocks**: Never assume `msg.content[0]` is a TextBlock. Filter by `block.type == "text"` — some providers return ThinkingBlock or other types first. Use `next((b for b in msg.content if b.type == "text"), None)` with a None default to avoid StopIteration in async contexts.
- **FTS5 input**: Always pass user query strings through `_escape_fts_query()` (db.py) before MATCH evaluation. FTS5 treats bare `OR`, `*`, `:`, `^` as operators — unescaped user input raises `OperationalError`.
- **ChromaDB client**: Only reach the vector store via `_get_collection()` (embed.py), which serializes lazy construction behind a lock. chromadb 1.x corrupts its global client state under concurrent cold-init (`RustBindingsAPI` / `tenant default_tenant` errors that wedge Chroma for the whole process); constructing a `PersistentClient` directly elsewhere reintroduces that race. Note `chunks_fts.content` is tokenized for BM25 — the FTS arm back-fills raw text from Chroma so rerank/render never see token-soup.

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
| `meta` | JSON blob: `{ video_id, list_id, title, cover_url, duration_seconds, uploader, source_url, platform, ui_lang }` (ingest); `{ source_id, list_id, source_title, ui_lang }` (digest rerun) |

### `sources` — active video catalog

| Column | Notes |
|---|---|
| `video_id` | Platform-native ID (e.g. `bvid`) |
| `title`, `summary`, `keywords` | `title` is platform metadata; `summary`/`keywords` columns DROPPED — sections are the sole digest store |
| `language`, `uploader`, `duration_seconds` | Video metadata |
| `whisper_model`, `ai_model` | Processing config at ingest time |
| `cover_url`, `processed_at`, `settings_snapshot` | Ingest-time state |
| `series_name`, `sequence_number`, `season_number` | Facet metadata extracted by LLM digest (nullable) |

### `artifacts` — Lab-generated content

| Column | Notes |
|---|---|
| `name` | User-visible name (initial = type label, renameable) |
| `type` | `"brief"` \| `"study_guide"` \| `"blog_post"` \| `"custom_report"` \| `"mind_map"` (JSON tree, viewer-rendered) \| `"chat_message"` (saved chat turn) |
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
| `status` | In-flight: user `"pending"`, assistant `"streaming"`. Terminal: `"done"` \| `"failed"` \| `"cancelled"`. Both rows of a turn flip to the **same** terminal status atomically (`update_turn_terminal`); a turn is visible to LLM replay + compaction iff both are `"done"`. UI history is unfiltered (renders 已停止/重试); the LLM snapshot filters to `"done"` inline (not in `get_recent_messages`). |
| `error` | Error code (e.g. `llm_rate_limit_error`, `internal_error`), nullable; set on producer failure or server restart sweep; mapped by `classify_error()` from SDK exceptions; frontend resolves via `chat.errors.*` i18n keys |
| `metadata` | JSON blob, nullable: `{"tool_calls": [...], "rag": {"calls": [...]}}` — full field shape in `docs/chat_architecture.md`. Set by `run_chat_turn` post-stream. No migration for legacy rows — the ledger renders best-effort from whatever fields exist. |

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

### `sections` — bounded sub-source spans

| Column | Notes |
|---|---|
| `source_id` | FK to `sources.id`, ON DELETE CASCADE |
| `seq` | Section order within a source |
| `seg_start` | First segment index in the source |
| `seg_end` | Last segment index in the source (inclusive) |
| `token_count` | Sum of `count_tokens(seg.text)` across the section's segments |
| `timestamp_start` | Wall-clock seconds at `segments[seg_start].start` |
| `timestamp_end` | Wall-clock seconds at `segments[seg_end].end` |
| `summary` | LLM refine-summary; always populated (written with the section row) |
| `keywords` | LLM keywords (JSON array); always populated |

Index: `idx_sections_source` on `(source_id, seq)`. Chunks produced per-section nest physically (a chunk's `[seg_start, seg_end]` is fully contained in exactly one section's). Production write path is `_exec_write_sections` (called from `write_source_with_segments` in the same transaction as source + segments); `section_digests` is required whenever `sections` is provided, so every section row carries a summary.

### `chunks_fts` — FTS5 virtual table

BM25-ranked full-text index over transcript chunks. Populated by `populate_fts()` during the embed stage; cleared per-source on re-ingest. Columns: `content`, `source_id`, `video_title`, `timestamp_start`, `timestamp_end`, `chunk_id`, `seg_start`, `seg_end`. Query via `query_fts_rows()` in `db.py`.

## Ingestion Pipeline

```
POST /ingest/url → resolve → dedup check → create job(s)
  → worker: download → audio → transcribe → punctuate → derive_sections → chunk (per-section) → digest ∥ embed → write_source + write_transcript_segments + write_sections (atomic)
```

- Dedup via `get_video_statuses` (sources + jobs); skip if processed or in-flight.
- Full re-process: DELETE /sources/:id then re-ingest.
- POST /sources/:id/rerun re-runs the digest section-by-section (`digest_sections` over the stored section rows; updates each section's summary/keywords + the source's facet columns); transcript and sections are reused, never re-derived. A source with 0 section rows fails loud (re-ingest).
- Delete removes ChromaDB embeddings and `sources` row (transcript segments cascade via FK)

### Pipeline stages (per video)

1. **download** → temp video file
2. **audio** → strip to .wav via FFmpeg; probes streams first — a track-less video fails loud (`video has no audio track`), never the raw FFmpeg dump (probe failure fails open to FFmpeg's own error)
3. **transcribe** → FunASR AutoModel (SenseVoice or Whisper via WhisperWarp) → raw VAD segments with timestamps + speaker labels
4. **punctuate** → ct-punc (zh-gated) → punctuated sentence segments persisted to `transcript_segments`; empty result (music-only / silent audio) fails loud (`no speech detected in audio`) — the cancel gate runs first, a user cancel wins over the failure
5. **derive_sections** → token+pause boundary, target=12000 (zone [7200, 16800]). Short videos = 1 section spanning all. Produces `sections` rows.
6. **chunk** → greedily merge consecutive **sentence** segments within each section to token target, split at trustworthy sentence boundary. Records `seg_start`/`seg_end` per chunk (source-global indices). Chunks physically nest in sections.
7. **digest** → `digest_sections`: section 1 via `digest()` (summary, keywords, facets — extracted once), sections 2..N via a refine prompt with rolling context. Per-section summary/keywords land on each `sections` row; facets land on `sources` via `apply_digest_facets`. 1-section sources are byte-identical to the pre-section path. (parallel with embed)
8. **embed** → store chunks in ChromaDB with per-source and per-list scope (parallel with digest). Chroma metadata keys on `source_id` (+ `seg_start`/`seg_end`), not `video_id`.
9. **write_source** → upsert source row + transcript_segments + sections atomically in one transaction (via `write_source_with_segments`)

## Chat Pipeline

Full mechanism — request lifecycle, retrieval pipeline, tool loop, SSE semantics, compression, eval endpoint, prompt-trace — lives in `docs/chat_architecture.md`. The rules below are what an AI changing this code must not break:

- `POST /lists/:id/chat` streams SSE via a producer/consumer split (`run_chat_turn` → `StreamBuffer` → `_sse_consumer`); both rows of a turn flip to the **same** terminal status atomically (`update_turn_terminal`) and `active_stream_message_id` clears in the same transaction — breaking that wedges the conversation at 409.
- SSE events: `meta`, `delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`. `meta` is always first (carries `{message_id}` for cancel-by-id, constant `SSE_EVENT_META`); `error` events carry machine-readable codes (via `classify_error()`) for frontend i18n.
- `stream_with_tools` is a bounded loop (max 3 iterations). Exhausted iterations keep tools **advertised** but unexecuted (gate is `is_synthesis_turn`) — withholding them makes native tool-call tokens leak into the visible answer. Any retrieve-family call disables all three tools for the rest of the turn. An empty synthesis gets one forced follow-up call; still no text → typed no-text error, never a blank persisted message.
- Retrieval: facet scoping fails open to the full pool; rerank order is final (no relevance gate); rerank/embed consume raw `chunk.content` while the LLM sees speaker-turn reconstruction — keep that split (embedder parity).
- Prior-turn tool exchanges are dropped from history replay (`expand_message_for_provider`); only synthesized prose survives — never assume prior evidence is still in LLM context.
- Compression summary is prose only — `[N]` markers deliberately dropped (see `docs/citation_system.md`).
- The eval endpoint (`POST /eval/run_chat`) shares the prod engine but mirrors two literals that must stay in sync when touching prod: the `"\n\n"` break around tool calls and the `"tool_error"` code. Its 422 detail stays messages-only — `str(e)` would leak `api_key` via the merged llm-override dump.

## Artifact pipeline

`_run_artifact_job` (worker.py) loads each selected source's sections via `_build_section_views`, reconstructs verbatim text per section with `format_turns(include_time=False)`, and calls `_refine_artifact`. When all sections fit in one batch (the common case — a few short sources, each with 1 section), `_refine_artifact` calls `_call_llm` exactly once with a prompt byte-identical to the legacy single-call template (regression guard: `tests/test_artifact_refine.py::test_refine_artifact_single_batch_byte_identical_prompt`). When sections don't fit, the running-draft refine path calls `_call_llm` once per batch: batch 1 produces an initial draft; batch k>1 feeds the running draft + new sections with an "integrate this new material" directive. Per-section failure (`token_count > budget`) and missing-sections failure (no `sections` rows for a source) both fail loud via `PipelineError`. Soft cost note: `logger.warning` when batch count > 3 (no schema/UI change).

## Platform Adapters

```python
class PlatformAdapter:
    def resolve_flat(url) -> PlaylistMeta
    async def get_videos_metadata(video_ids) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]
    def download(video_id, source_url, connections) -> Path
```

**Registry dispatch** (`adapters/__init__.py`): `get_adapter_for_url` (domain suffix match, resolve path) and `get_adapter_for_platform` (metadata + worker paths; job meta carries `platform`; `VideoMetadataRequest.platform` is required, no default). Unknown target → `UnsupportedPlatformError` → 400. `CDN_DOMAINS` beside the registry maps each platform to its cover-CDN hosts + optional Referer; `routers/proxy.py` derives its allowlist and Referer policy from it (an import-time assert keeps the two key sets equal — a new platform can't land with working ingest but broken covers). Shared yt-dlp plumbing lives in `adapters/_ytdlp_common.py`: `strip_ansi`, `apply_aria2c`, `pick_thumbnail` (max-by-area, never list order), `safe_duration` (non-numeric → 0, never sinks the list), `gather_metadata` (thread-pool per-id fetch, failed ids omitted), `raise_mapped` (auth regex → `AuthRequiredError`, overrides, hint). `resolve_flat` is blocking yt-dlp — the router runs it via `asyncio.to_thread` so the event loop stays responsive.

- **bilibili** — single / multi-part (`?p=N`) / collection / favorite lists; courses raise `AuthRequiredError`. Cookie auth (QR login). 403/412 → `needs_auth`. Keeps its own inline error mapping — lowercased matching, 412 handling and cookie revalidation don't fit `raise_mapped`.
- **youtube** — single videos + public playlists, no credentials; sign-in/age/private/members-only messages → `AuthRequiredError`. Flat playlist entries carry full metadata, so phase-2 enrichment is a light per-id refetch.
- **tiktok** — single videos (incl. `vm.`/`vt.`/`/t/` short links) + collections, no credentials; **best-effort tier** (extraction breaks in waves; generic failures append an upgrade-yt-dlp hint). Captions are bounded to 120 chars as titles; image posts → named `DownloadError`; the `yt-dlp[curl-cffi]` extra supplies the TLS impersonation the extractor requests. Download filters formats by `vcodec` (h264 preferred), never `acodec` — the extractor stamps a fabricated `acodec` on every format, and TikTok's HEVC (`bytevc1`) variants are silent files.

**Two-phase resolve** (bilibili): `resolve_flat` enumerates fast via `extract_flat="in_playlist"` — keep it flat; a multi-part video comes back as a flat playlist of `?p=N` parts (no per-part title/duration), and non-flat extraction re-resolves every part's stream formats (~15× slower) for data phase 2 supplies anyway. `get_videos_metadata` then enriches each part from the bilibili view API (per-part title/duration) and expands a bare multi-part BVID into `_pN` parts. Part title is the composite `"<part> - <video>"` (part-first so it survives single-line truncation while the parent title trails as collection context).

**QR login flow**: `POST /auth/bilibili/qr` → get `{url, key}` → UI polls `GET /auth/bilibili/qr/status?key=...` (query param, not path param — avoids key in server logs) → on success, cookie saved to config.

**Cookie file**: `_cookie_file()` converts the raw cookie string to Netscape HTTP Cookie File format (yt-dlp requirement). A module-level `_cookie_file_cache` skips the disk write when the cookie string is unchanged.

## Configuration Schema

```json
{
  "accounts": { "bilibili": { "cookie": "", "username": "", "avatar_url": "" } },
  "ai": { "protocol": "openai|anthropic", "model": "", "api_key": "", "base_url": null, "output_language": "ui", "context_window": 128000, "max_output_tokens": 16384 },
  "transcription": { "model": "sensevoice-small|large-v3", "device": "cuda|cpu", "language": "auto" },
  "backend": { "port": 8765, "max_concurrent_jobs": 1, "cors_origins": [...] },
  "rag": { "max_distance": 0.8, "reranking_enabled": true, "hybrid_enabled": true, "debug_prompts": false }
}
```
Reranker is the single spec `RERANKER_SPEC_ID = "bge-reranker-base-q"` (int8, zh+en); its ONNX session must source providers from `interpreting_providers()` — on macOS that excludes CoreML, whose per-input-shape JIT recompile hangs >90s / OOM-kills the reranker on first chat retrieve. `FIND_PASSAGES_TOP_K = 8`. Model rationale and prompt-trace details: `docs/chat_architecture.md`.
