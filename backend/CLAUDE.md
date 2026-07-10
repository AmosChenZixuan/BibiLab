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
  chat.py           /lists/:id/chat (SSE + cancel + reattach), /lists/:id/conversation, /debug/messages/:msg_id (debug_router); stream_with_tools loop; _classify_llm_error
  eval.py           /eval/run_chat (stateless one-shot chat for eval frameworks), /eval/llm (bare _call_llm passthrough) — see docs/chat_architecture.md
  lists.py          /lists/* (CRUD)
  ingest.py         /ingest/url, /ingest/preview, /ingest/preview/metadata
  sources.py        /sources/* (source content, covers, sections list, rerun, PATCH facets manual edit)
  artifacts.py      /lists/:id/artifacts (list + create), /artifacts/:id (+ /content)
  jobs.py           /jobs, /jobs/:id (GET + DELETE)
  models.py         model registry API — unified listing + download for local model deps
  config_router.py  /config (GET + PUT)
  health.py         /health
  proxy.py          /proxy/cover — cover-CDN proxy; allowlist + Referer derived from adapters CDN_DOMAINS
  _model_gate.py    pre-flight 412 gate shared by ingest/chat/artifacts routers
models/           — Pydantic request/response models + domain errors (chat.py, ingest.py, artifacts.py, jobs.py, models.py, _enums.py)
pipeline/         — one file per stage
  _shared.py        sync _call_llm + async stream_llm (OpenAI/Anthropic), StreamEvent/ToolCall/ToolDefinition dataclasses
  audio.py          FFmpeg audio extraction (video → .wav)
  transcribe.py     FunASR AutoModel (SenseVoice/Whisper) + CAM++ diarization → VAD segments w/ speaker labels
  chunk.py          per-section greedy segment merger → token target (`zh=800`, `en=300`); physical chunk `[seg_start, seg_end]` is fully contained in one section's range
  section.py        Section dataclass + derive_sections (token+pause boundary, target=12000) + chunk_by_sections (per-section chunking with source-global re-stamp)
  digest.py         LLM digest: per-section summary/keywords → sections table; facets → sources
  embed.py          ChromaDB embed + retrieve() (hybrid search → rerank → aggregation), FTS5 populate
  rerank.py         lazy ONNX cross-encoder reranker (single spec RERANKER_SPEC_ID; providers via interpreting_providers(), CoreML excluded)
  chat_tools.py     find_passages + read_section definitions, execution dispatcher, section fencing, CitationRegistry, `_NO_MATCH_NOTE`
  chat_summary.py   conversation compression (sliding window + LLM summary; summary is prose only — [N] markers not preserved)
  citation_parser.py incremental citation parser — strips [N] tokens from LLM deltas, emits citation SSE events with {index, section_id, source_id, timestamp_start, chunk_ids}
  chat_runs.py       StreamBuffer + ChatRunRegistry; in-memory buffer decouples LLM producer from HTTP request lifetime
  chat_inference_pool.py dedicated ThreadPoolExecutor for chat-path inference (rerank + Chroma query), isolated from the default executor
  punctuate.py      ct-punc zh-gated punctuation; alignment failure falls back to unpunctuated (never fatal)
adapters/         — platform download + resolution; __init__ holds the registry (URL/platform dispatch) + CDN_DOMAINS; base.py dataclasses/interface; _ytdlp_common shared plumbing; bilibili / youtube / tiktok implementations
db.py             — SQLite schema + query helpers
video_status.py   — derive_video_statuses (status mapping extracted from db.py per Code Health Rule #4)
config.py         — settings persisted to ~/.bibilab/config.json; includes models_dir() helper
worker.py         — SQLite-polling job dispatcher; accepts config/adapter/home via constructor for testability
cleanup.py        — resource cleanup utilities
model_registry.py — all non-LLM model downloads via ensure() (per-model locks, atomic .partial→rename); holds RERANKER_SPEC_ID
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

Column-level reference: `docs/data_model.md`; authoritative DDL: `db.py` (`bootstrap_db`). Per-table invariants that must hold:

- `jobs` — ephemeral queue (`queued → downloading → transcribing → processing → done | failed | needs_auth`); `sources` is the dedup source of truth, never `jobs`.
- `sources` — no `summary`/`keywords` columns; sections are the sole digest store, `sources` carries only the LLM facet columns (`series_name`, `sequence_number`, `season_number`, nullable).
- `messages.status` — both rows of a turn flip to the **same** terminal status atomically (`update_turn_terminal`); a turn is visible to LLM replay + compaction iff both are `"done"`; UI history is unfiltered, the LLM snapshot filters to `"done"` inline (not in `get_recent_messages`).
- `messages.error` — machine-readable code (see SSE rules under Chat Pipeline); `messages.metadata` renders best-effort, no migration for legacy rows.
- `conversations` — one per list (UNIQUE FK), auto-created on first message; `active_stream_message_id` must be cleared in the same transaction as the terminal flip.
- `transcript_segments` / `sections` — written atomically with the source row (`write_source_with_segments`); `section_digests` is required whenever `sections` is provided, so every section row carries a summary; chunks nest physically in exactly one section.
- `chunks_fts` — FTS5, populated by `populate_fts()` at embed, cleared per-source on re-ingest; query only via `query_fts_rows()` (which escapes MATCH input).

## Ingestion Pipeline

```
POST /ingest/url → resolve → dedup check → create job(s)
  → worker: download → audio → transcribe → punctuate → derive_sections → chunk (per-section) → digest ∥ embed → write_source + write_transcript_segments + write_sections (atomic)
```

Stage-by-stage mechanism (fail-loud contracts, duration validation, two-phase resolve) lives in `docs/ingestion_architecture.md`. Rules that must hold:

- Dedup source of truth is `sources` (via `get_video_statuses`); a video is "processed" iff it has a `sources` row. Full re-process = DELETE + re-ingest.
- Source + transcript_segments + sections land **atomically in one transaction** (`write_source_with_segments`); digest rerun reuses stored sections, never re-derives — 0 section rows fails loud.
- Section/chunk invariants: sections target=12000 tokens (zone [7200, 16800]), ≥1 per source; chunk token target `zh=800` / `en=300`; a chunk's `[seg_start, seg_end]` nests in exactly one section (source-global indices).
- Every stage failure surfaces on the job row as `[<stage>] <message>` — keep new failure paths behind clear messages, not raw tool dumps; the cancel gate runs before fail-loud guards (a user cancel wins).
- punctuate is zh-gated and never fatal (alignment failure falls back to unpunctuated); digest ∥ embed run in parallel; Chroma metadata keys on `source_id`, not `video_id`.

## Chat Pipeline

Full mechanism — request lifecycle, retrieval pipeline, tool loop, SSE semantics, compression, eval endpoint, prompt-trace — lives in `docs/chat_architecture.md`. The rules below are what an AI changing this code must not break:

- `POST /lists/:id/chat` streams SSE via a producer/consumer split (`run_chat_turn` → `StreamBuffer` → `_sse_consumer`); the turn's terminal flip is atomic (see Database Schema invariants) — breaking it wedges the conversation at 409.
- SSE events: `meta`, `delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`. `meta` is always first (carries `{message_id}` for cancel-by-id, constant `SSE_EVENT_META`); `error` events carry machine-readable codes (via `_classify_llm_error()`) for frontend i18n.
- `stream_with_tools` is a bounded loop (max 3 iterations). Exhausted iterations keep tools **advertised** but unexecuted (gate is `is_synthesis_turn`) — withholding them makes native tool-call tokens leak into the visible answer. Any retrieve-family call disables all three tools for the rest of the turn. An empty synthesis gets one forced follow-up call; still no text → typed no-text error, never a blank persisted message.
- Retrieval: facet scoping fails open to the full pool; rerank order is final (no relevance gate); rerank/embed consume raw `chunk.content` while the LLM sees speaker-turn reconstruction — keep that split (embedder parity).
- Prior-turn tool exchanges are dropped from history replay (`expand_message_for_provider`); only synthesized prose survives — never assume prior evidence is still in LLM context.
- Compression summary is prose only — `[N]` markers deliberately dropped (see `docs/citation_system.md`).
- The eval endpoint (`POST /eval/run_chat`) shares the prod engine but mirrors two literals that must stay in sync when touching prod: the `"\n\n"` break around tool calls and the `"tool_error"` code. Its 422 detail stays messages-only — `str(e)` would leak `api_key` via the merged llm-override dump.

## Artifact pipeline

Mechanism in `docs/ingestion_architecture.md`. Rules: the single-batch prompt stays **byte-identical** to the legacy template (regression guard `tests/test_artifact_refine.py::test_refine_artifact_single_batch_byte_identical_prompt`); over-budget sections and missing sections fail loud via `PipelineError` — no silent truncation.

## Platform Adapters

```python
class PlatformAdapter:
    def resolve_flat(url) -> PlaylistMeta
    async def get_videos_metadata(video_ids) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]
    def download(video_id, source_url, connections) -> Path
```

Registry dispatch, per-platform behavior, two-phase resolve, and bilibili auth are documented in `docs/ingestion_architecture.md`. Rules that must hold:

- Adding a platform: register the adapter **and** its `CDN_DOMAINS` entry — an import-time assert keeps the two key sets equal (ingest can't land with broken covers). `VideoMetadataRequest.platform` stays required, no default.
- Reuse `adapters/_ytdlp_common.py` (`strip_ansi`, `apply_aria2c`, `pick_thumbnail`, `safe_duration`, `gather_metadata`, `raise_mapped`) before writing per-adapter plumbing; bilibili's inline error mapping is the sanctioned exception.
- `resolve_flat` is blocking yt-dlp — call it via `asyncio.to_thread`, and keep it flat (`extract_flat="in_playlist"`): non-flat extraction is ~15× slower for data phase 2 supplies anyway.
- tiktok download filters formats by `vcodec` (h264 preferred), never `acodec` — the extractor fabricates `acodec`, and TikTok's HEVC (`bytevc1`) variants are silent files.

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
