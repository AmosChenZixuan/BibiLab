# Ingestion & Adapter Architecture

How a video becomes a searchable source: URL resolution, platform adapters, the worker pipeline, and the artifact generator. The rules an AI must follow when *changing* this code live in `backend/CLAUDE.md`; this document explains how the system behaves. Chat/retrieval is covered in `docs/chat_architecture.md`.

## Ingestion flow

```
POST /ingest/url → resolve → dedup check → create job(s)
  → worker: download → audio → transcribe → punctuate → derive_sections → chunk (per-section) → digest ∥ embed → write_source + write_transcript_segments + write_sections (atomic)
```

- Dedup via `get_video_statuses` (sources + jobs); skip if processed or in-flight. `sources` is the dedup source of truth; `jobs` is ephemeral — a video is "processed" iff it has a `sources` row.
- Full re-process: `DELETE /sources/:id` then re-ingest.
- `POST /sources/:id/rerun` re-runs the digest section-by-section (`digest_sections` over the stored section rows; updates each section's summary/keywords + the source's facet columns); transcript and sections are reused, never re-derived. A source with 0 section rows fails loud (re-ingest).
- Delete removes ChromaDB embeddings and the `sources` row (transcript segments cascade via FK).

## Pipeline stages (per video)

1. **download** → temp video file, via the platform adapter.
2. **audio** → strip to 16kHz mono .wav via FFmpeg. Probes streams first: a track-less video fails loud (`video has no audio track`) instead of surfacing FFmpeg's raw "does not contain any stream" dump; a probe failure fails open so an unreadable file still gets FFmpeg's own error. The decoded wav duration is validated against two references — the input container's duration and the platform-reported duration (either may be 0 = unknown; both unknown logs an "unverified" warning) — because a byte-truncated faststart m4a still decodes to a short wav with ffmpeg exiting 0.
3. **transcribe** → FunASR AutoModel (SenseVoice or Whisper via WhisperWarp) + CAM++ diarization → raw VAD segments with timestamps + speaker labels.
4. **punctuate** → ct-punc, gated to `zh` (char-offset alignment strips spaces — correct for CJK, destructive for English; the false-。 VAD defect is CJK-specific). Alignment failure falls back to unpunctuated segments — punctuation is an enhancement, never fatal. Sentence segments persist to `transcript_segments`. An empty result (music-only / silent audio) fails loud (`no speech detected in audio`); the cancel gate runs first so a user cancel wins over the failure.
5. **derive_sections** → token+pause boundary, target=12000 (zone [7200, 16800]). Short videos = 1 section spanning all. Produces `sections` rows.
6. **chunk** → greedily merge consecutive **sentence** segments within each section to a token target (`zh=800`, `en=300`), split at trustworthy sentence boundaries. Records `seg_start`/`seg_end` per chunk (source-global indices). Chunks physically nest in sections — a chunk's `[seg_start, seg_end]` is fully contained in exactly one section's range.
7. **digest** → `digest_sections`: section 1 via `digest()` (summary, keywords, facets — extracted once), sections 2..N via a refine prompt with rolling context. Per-section summary/keywords land on each `sections` row; facets land on `sources` via `apply_digest_facets`. 1-section sources are byte-identical to the pre-section digest path. Runs in parallel with embed.
8. **embed** → store chunks in ChromaDB with per-source and per-list scope. Chroma metadata keys on `source_id` (+ `seg_start`/`seg_end`), not `video_id`. Runs in parallel with digest.
9. **write_source** → upsert source row + transcript_segments + sections atomically in one transaction (`write_source_with_segments`).

Stage failures surface on the job row as `[<stage>] <message>` — the worker wraps each stage's exceptions with a stage prefix (`[downloading]`, `[transcribing]`, `[processing]`, `[persisting]`).

## Platform adapters

```python
class PlatformAdapter:
    def resolve_flat(url) -> PlaylistMeta
    async def get_videos_metadata(video_ids) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]
    def download(video_id, source_url, connections) -> Path
```

### Registry dispatch (`adapters/__init__.py`)

`get_adapter_for_url` (domain suffix match, resolve path) and `get_adapter_for_platform` (metadata + worker paths; job meta carries `platform`; `VideoMetadataRequest.platform` is required, no default — a silent bilibili assumption would misroute every other platform). Unknown target → `UnsupportedPlatformError` → 400.

`CDN_DOMAINS` beside the registry maps each platform to its cover-CDN hosts + optional Referer; `routers/proxy.py` derives its allowlist and Referer policy from it, and an import-time assert keeps the two key sets equal — a new platform can't land with working ingest but broken covers.

Shared yt-dlp plumbing lives in `adapters/_ytdlp_common.py`: `strip_ansi`, `apply_aria2c`, `pick_thumbnail` (max-by-area, never list order), `safe_duration` (non-numeric → 0, never sinks the list), `gather_metadata` (thread-pool per-id fetch, failed ids omitted), `raise_mapped` (auth regex → `AuthRequiredError`, overrides, hint).

`resolve_flat` is blocking yt-dlp — the router runs it via `asyncio.to_thread` so the event loop stays responsive.

### Per-platform behavior

- **bilibili** — single / multi-part (`?p=N`) / collection / favorite lists; courses raise `AuthRequiredError`. Cookie auth (QR login). 403/412 → `needs_auth`. Keeps its own inline error mapping — lowercased matching, 412 handling and cookie revalidation don't fit `raise_mapped`.
- **youtube** — single videos + public playlists, no credentials; sign-in/age/private/members-only messages → `AuthRequiredError`. Flat playlist entries carry full metadata, so phase-2 enrichment is a light per-id refetch.
- **tiktok** — single videos (incl. `vm.`/`vt.`/`/t/` short links) + collections, no credentials; **best-effort tier** (extraction breaks in waves; generic failures append an upgrade-yt-dlp hint). Captions are bounded to 120 chars as titles; image posts → a named `DownloadError`; the `yt-dlp[curl-cffi]` extra supplies the TLS impersonation the extractor requests. Download filters formats by `vcodec` (h264 preferred), never `acodec` — the extractor stamps a fabricated `acodec` on every format, and TikTok's HEVC (`bytevc1`) variants are silent files.

### Two-phase resolve (bilibili)

`resolve_flat` enumerates fast via `extract_flat="in_playlist"` — keep it flat: a multi-part video comes back as a flat playlist of `?p=N` parts (no per-part title/duration), and non-flat extraction re-resolves every part's stream formats (~15× slower) for data phase 2 supplies anyway. `get_videos_metadata` then enriches each part from the bilibili view API (per-part title/duration) and expands a bare multi-part BVID into `_pN` parts. Part title is the composite `"<part> - <video>"` — part-first so it survives single-line truncation while the parent title trails as collection context.

### Auth (bilibili)

- **QR login flow**: `POST /auth/bilibili/qr` → get `{url, key}` → UI polls `GET /auth/bilibili/qr/status?key=...` (query param, not path param — avoids the key landing in server logs) → on success, cookie saved to config.
- **Cookie file**: `_cookie_file()` converts the raw cookie string to Netscape HTTP Cookie File format (yt-dlp requirement). A module-level `_cookie_file_cache` skips the disk write when the cookie string is unchanged.

## Artifact pipeline

`_run_artifact_job` (worker.py) loads each selected source's sections via `_build_section_views`, reconstructs verbatim text per section with `format_turns(include_time=False)`, and calls `_refine_artifact`.

- When all sections fit in one batch (the common case — a few short sources, each with 1 section), `_refine_artifact` calls `_call_llm` exactly once with a prompt byte-identical to the legacy single-call template (regression guard: `tests/test_artifact_refine.py::test_refine_artifact_single_batch_byte_identical_prompt`).
- When sections don't fit, the running-draft refine path calls `_call_llm` once per batch: batch 1 produces an initial draft; batch k>1 feeds the running draft + new sections with an "integrate this new material" directive.
- Per-section failure (`token_count > budget`) and missing-sections failure (no `sections` rows for a source) both fail loud via `PipelineError`.
- Soft cost note: `logger.warning` when batch count > 3 (no schema/UI change).
