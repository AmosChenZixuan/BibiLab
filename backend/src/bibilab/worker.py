"""Background worker loop."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from bibilab.adapters.base import AuthRequiredError, VideoMeta
from bibilab.cleanup import cleanup_job_artifacts, purge_download_files
from bibilab.config import BibilabConfig, bibilab_home, load_config
from bibilab.db import (
    apply_digest_facets,
    create_artifact,
    delete_job,
    get_list,
    get_pending_jobs,
    get_sections,
    get_source,
    get_transcript_segments,
    parse_job_meta,
    rows_to_sections,
    rows_to_segments,
    update_job_meta,
    update_job_status,
    update_section_summaries,
    write_source_with_segments,
)
from bibilab.model_registry import ensure
from bibilab.models.jobs import JobStatus, JobType
from bibilab.pipeline.artifact_refine import (
    _MIND_MAP_PROMPT,
    _build_section_views,
    _refine_artifact,
    _refine_mind_map,
    _render_mind_map_markdown,
)
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.digest import DigestResult, SectionDigest, digest_sections
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.punctuate import punctuate
from bibilab.pipeline.section import Section, chunk_by_sections, derive_sections, section_texts
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    transcribe,
)

logger = logging.getLogger(__name__)


async def reset_stuck_jobs() -> None:
    from bibilab.db import _in_placeholders, _now, get_db
    from bibilab.models.jobs import ACTIVE_JOB_STATUSES, JobStatus

    stuck = tuple(s for s in ACTIVE_JOB_STATUSES if s is not JobStatus.QUEUED)
    placeholders = _in_placeholders(list(stuck))
    async with get_db() as db:
        await db.execute(
            f"""
            UPDATE jobs SET status=?, updated_at=?
            WHERE status IN ({placeholders})
            """,
            (JobStatus.QUEUED, _now(), *stuck),
        )
        await db.commit()


def _reraise_gathered_failures(digest_raw: Any, embed_raw: Any) -> None:
    """Re-raise failures from the parallel digest∥embed gather in priority order.

    Digest is the primary error the job surfaces. When both failed, embed is
    logged as secondary and digest is raised; when only embed failed it
    propagates as the primary error with no misleading 'secondary' log."""
    digest_failed = isinstance(digest_raw, BaseException)
    embed_failed = isinstance(embed_raw, BaseException)
    if digest_failed and embed_failed:
        logger.error("embed_chunks also failed (secondary to the digest error)", exc_info=embed_raw)
    if digest_failed:
        raise digest_raw
    if embed_failed:
        raise embed_raw


def _download_cover(cover_url: str, dest: Path) -> bool:
    """Download cover image from URL to dest path. Returns True on success."""
    try:
        resp = httpx.get(cover_url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Job: failed to download cover from %s: %s", cover_url, exc)
        return False


class WorkerLoop:
    def __init__(
        self,
        concurrency: int = 1,
        config: BibilabConfig | None = None,
        adapter: Any = None,
        home: Path | None = None,
    ) -> None:
        self._config = config
        self._adapter = adapter
        self._bibilab_home = home if home is not None else bibilab_home()
        self._concurrency = concurrency
        self._task: asyncio.Task | None = None
        self._running = False
        self._cancelled: set[str] = set()
        self._in_flight: set[str] = set()

    async def start(self) -> None:
        await reset_stuck_jobs()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="bibilab-worker")
        logger.info("Worker loop started (concurrency=%d)", self._concurrency)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Worker loop stopped")

    def cancel_job(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    def _get_adapter(self, platform: str) -> Any:
        # ponytail: injected test seam wins regardless of platform; key the
        # seam by platform if a test ever needs two adapters at once.
        if self._adapter is not None:
            return self._adapter
        from bibilab.adapters import get_adapter_for_platform

        return get_adapter_for_platform(platform, self._get_config())

    def _get_config(self) -> BibilabConfig:
        if self._config is not None:
            return self._config
        return load_config()

    async def _loop(self) -> None:
        while self._running:
            if len(self._in_flight) < self._concurrency:
                pending = [dict(j) for j in await get_pending_jobs()]
                queued = [
                    j for j in pending if j["status"] == JobStatus.QUEUED.value and j["id"] not in self._in_flight
                ]
                slots = self._concurrency - len(self._in_flight)
                for job in queued[:slots]:
                    self._in_flight.add(job["id"])
                    asyncio.create_task(self._run_job(job))
            await asyncio.sleep(2)

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        try:
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            if job["type"] == JobType.ARTIFACT:
                await self._run_artifact_job(job)
                return

            if job["type"] == JobType.MODEL_DOWNLOAD:
                await self._download_model_job(job)
                return

            if job["type"] == JobType.DIGEST:
                await self._run_digest_job(job)
                return

            await self._pipeline(job)

        except AuthRequiredError as exc:
            logger.warning("Job %s needs auth (%s)", job_id, exc.resource_type)
            await update_job_status(job_id, JobStatus.NEEDS_AUTH.value, error=exc.resource_type)
        except asyncio.CancelledError:
            # Re-raise CancelledError - don't swallow it as a job failure
            raise
        except PipelineError as exc:
            logger.exception("Job %s pipeline error: %s", job_id, exc)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        except Exception as exc:
            logger.exception("Job %s (type=%s) failed", job_id, job.get("type", "unknown"))
            if job.get("type") in (None, JobType.INGEST):
                await asyncio.to_thread(cleanup_job_artifacts, job)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        finally:
            self._in_flight.discard(job_id)
            self._cancelled.discard(job_id)

    async def _download_model_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        spec_id = meta_raw.get("model_name", "")

        if not spec_id:
            logger.error("Model download job %s missing model_name in meta", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="missing model_name in job meta")
            return

        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        await asyncio.to_thread(ensure, spec_id)
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Model download job %s completed for %s", job_id, spec_id)

    # -------------------------------------------------------------------------
    # Artifact generation job
    # -------------------------------------------------------------------------

    async def _run_artifact_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        artifact_id = meta_raw["artifact_id"]
        list_id = meta_raw["list_id"]
        artifact_type = meta_raw["type"]
        prompt = meta_raw["prompt"]
        source_ids = meta_raw["source_ids"]
        cfg = self._get_config()
        # Mind-map jobs ignore the user-supplied prompt and use
        # `_MIND_MAP_PROMPT` instead. The rebind also feeds
        # `create_artifact`'s `prompt` column so the view-prompt modal
        # shows what the LLM actually saw.
        is_mind_map = artifact_type == "mind_map"
        if is_mind_map:
            prompt = _MIND_MAP_PROMPT

        try:
            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

            # Cancellation check before LLM work.
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=30)

            # _build_section_views handles source existence + no-sections
            # fail-loud.
            sections = await _build_section_views(source_ids)
            if is_mind_map:
                mind_map_result = await _refine_mind_map(
                    sections=sections,
                    cfg=cfg,
                    ui_lang=meta_raw.get("ui_lang"),
                )
                content = _render_mind_map_markdown(mind_map_result)
                artifact_name = mind_map_result.name
            else:
                artifact_result = await _refine_artifact(
                    prompt=prompt,
                    sections=sections,
                    cfg=cfg,
                    ui_lang=meta_raw.get("ui_lang"),
                )
                content = artifact_result.content
                artifact_name = artifact_result.name

            # Check for cancellation before writing file
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

            # Write artifact content to file
            artifacts_dir = self._bibilab_home / "artifacts" / list_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            content_path = artifacts_dir / f"{artifact_id}.md"
            content_path.write_text(content, encoding="utf-8")

            # Create artifact record with success status
            await create_artifact(
                artifact_id=artifact_id,
                list_id=list_id,
                name=artifact_name,
                type=artifact_type,
                prompt=prompt,
                source_ids=source_ids,
                status="completed",
                content_path=str(content_path.relative_to(self._bibilab_home)),
                error=None,
            )

            await update_job_status(job_id, JobStatus.DONE.value, progress=100)
            logger.info("Artifact job %s completed for artifact %s", job_id, artifact_id)

        except PipelineError as exc:
            error_message = str(exc)
            logger.error("Artifact job %s failed: %s", job_id, error_message)
            await update_job_status(job_id, JobStatus.FAILED.value, error=error_message)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    async def _call_digest_sections(
        self,
        section_texts_list: list[str],
        video_meta: VideoMeta,
        cfg: BibilabConfig,
        ui_lang: str | None = None,
    ) -> tuple[DigestResult, list[SectionDigest]]:
        """Run the section-level digest. Does not catch exceptions — callers handle errors."""
        return await asyncio.to_thread(
            digest_sections,
            section_texts_list,
            video_meta,
            cfg.ai,
            cfg.ai.output_language,
            ui_lang,
        )

    # -------------------------------------------------------------------------
    # Digest-only job (rerun)
    # -------------------------------------------------------------------------

    async def _run_digest_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        source_id: str = meta_raw.get("source_id", "")
        cfg = self._get_config()

        source = await get_source(source_id)
        if source is None:
            logger.warning("Digest job %s: source %s not found", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=f"Source {source_id!r} not found")
            return

        segments = await get_transcript_segments(source_id)
        if not segments:
            logger.warning("Digest job %s: source %s has no transcript", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="Source has no transcript")
            return

        whisper_segments = rows_to_segments(segments)

        section_rows = await get_sections(source_id)
        if not section_rows:
            # Defensive: unreachable through the normal ingest path (sections
            # are written atomically with the source). Surface a re-ingest hint
            # if it ever happens.
            msg = "Source has no sections; re-ingest the source to derive them"
            logger.error("Digest job %s: %s", job_id, msg)
            await update_job_status(job_id, JobStatus.FAILED.value, error=msg)
            return

        sections = rows_to_sections(section_rows)
        section_texts_list = section_texts(whisper_segments, sections)
        video_meta = VideoMeta.from_source(source)

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

        try:
            extraction, section_digests = await self._call_digest_sections(
                section_texts_list, video_meta, cfg, meta_raw.get("ui_lang")
            )
        except Exception as exc:
            logger.exception("Digest job %s failed", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
            return

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

        # Two writes: per-section summaries, then the source's facet columns.
        # No transaction needed — they touch disjoint rows/columns and a
        # rerun is idempotent (re-running fixes any partial write).
        await update_section_summaries(
            source_id,
            [(row["seq"], sd.summary, sd.keywords) for row, sd in zip(section_rows, section_digests)],
        )
        await apply_digest_facets(
            source_id,
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
            bump_processed_at=False,
        )

        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Digest job %s completed for source %s", job_id, source_id)

    # -------------------------------------------------------------------------
    # Pipeline stages (full ingest)
    # -------------------------------------------------------------------------

    async def _pipeline(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = parse_job_meta(job)
        cfg = self._get_config()

        source_id: str = meta_raw.get("source_id") or str(uuid.uuid4())
        meta_raw["source_id"] = source_id
        job["meta"] = meta_raw  # cleanup snapshot must carry source_id (Q1)
        await update_job_meta(job_id, {"source_id": source_id})
        video_id: str = meta_raw["video_id"]
        list_id: str = meta_raw["list_id"]

        list_row = await get_list(list_id)
        if list_row is None:
            raise PipelineError(f"List {list_id!r} not found - it may have been deleted")

        video_meta = VideoMeta(
            video_id=video_id,
            title=meta_raw.get("title", ""),
            # Jobs queued before multi-platform routing carry no platform (or an
            # empty string from old UI retry payloads); they can only have come
            # from the bilibili-only ingest path.
            platform=meta_raw.get("platform") or "bilibili",
            source_url=meta_raw.get("source_url", ""),
            cover_url=meta_raw.get("cover_url", ""),
            duration_seconds=meta_raw.get("duration_seconds", 0),
            uploader=meta_raw.get("uploader", ""),
        )

        # Stage 1: Download
        try:
            video_path = await self._stage_download(job, video_meta, source_id)
        except Exception as exc:
            raise PipelineError(f"[downloading] {exc}") from exc
        if video_path is None:
            return  # cancelled

        # Stage 2: Audio extraction
        try:
            wav_path = await self._stage_extract_audio(job, video_path, video_meta.duration_seconds)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if wav_path is None:
            return  # cancelled

        # Stage 3: Transcription
        try:
            result = await self._stage_transcribe(job, wav_path, source_id, cfg)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if result is None:
            return  # cancelled
        detected_language, effective_language, sentence_segments = result

        # Stage 4: Derive sections, chunk per-section, digest + embed in parallel
        try:
            result = await self._stage_process(
                job, sentence_segments, source_id, video_meta, list_id, cfg, effective_language
            )
        except Exception as exc:
            raise PipelineError(f"[processing] {exc}") from exc
        if result is None:
            return  # cancelled
        extraction, sections, section_digests = result

        # Stage 5: Persist source + segments + sections atomically, then cleanup
        try:
            await self._stage_persist(
                job_id,
                source_id,
                video_id,
                video_meta,
                list_id,
                extraction,
                sections,
                section_digests,
                detected_language,
                cfg,
                sentence_segments,
            )
        except Exception as exc:
            raise PipelineError(f"[persisting] {exc}") from exc
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Job %s completed for video %s", job_id, video_id)

    async def _abort_if_cancelled(self, job: dict) -> bool:
        """If the job is cancelled, purge its artifacts, delete it, and report
        True so the caller stops. Safe to call more than once within a stage.

        Caller must pass the full job dict: cleanup_job_artifacts is a no-op
        unless the job has type "ingest" and a parseable meta.video_id, so an
        id-only snapshot would purge nothing. Discard the tracking flag only
        after the delete lands — if cleanup or delete raises, the flag stays
        set and the row is still marked failed by _run_job's handler."""
        job_id = job["id"]
        if job_id not in self._cancelled:
            return False
        await asyncio.to_thread(cleanup_job_artifacts, job)
        await delete_job(job_id)
        self._cancelled.discard(job_id)
        return True

    async def _stage_download(
        self,
        job: dict,
        video_meta: VideoMeta,
        source_id: str,
    ) -> Path | None:
        """Stage 1: Download video file and cover image."""
        await update_job_status(job["id"], JobStatus.DOWNLOADING.value, progress=10)
        if await self._abort_if_cancelled(job):
            return None

        # .part hygiene: purge any downloads/{id}.* left by a prior failed/corrupt
        # attempt so this download starts clean and never resumes onto stale bytes.
        await asyncio.to_thread(purge_download_files, video_meta.video_id)

        if await self._abort_if_cancelled(job):
            return None
        video_path: Path = await asyncio.to_thread(
            self._get_adapter(video_meta.platform).download,
            video_meta.video_id,
            video_meta.source_url,
            self._get_config().backend.download_connections,
        )

        # Download cover
        covers_dir = self._bibilab_home / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        cover_dest = covers_dir / f"{source_id}.jpg"
        await asyncio.to_thread(_download_cover, video_meta.cover_url, cover_dest)

        return video_path

    async def _stage_extract_audio(
        self,
        job: dict,
        video_path: Path,
        expected_duration: float = 0.0,
    ) -> Path | None:
        """Stage 2: Extract audio from downloaded video."""
        await update_job_status(job["id"], JobStatus.TRANSCRIBING.value, progress=25)
        # On failure video_path is left on disk; cancel_or_delete_job cleans it up
        # later from the full job dict in the DB.
        wav_path = await asyncio.to_thread(extract_audio, video_path, expected_duration)

        if await self._abort_if_cancelled(job):
            return None

        return wav_path

    async def _stage_transcribe(
        self,
        job: dict,
        wav_path: Path,
        source_id: str,
        cfg: BibilabConfig,
    ) -> tuple | None:
        """Stage 3: Transcribe audio, punctuate (zh-gated) into sentence segments."""
        await update_job_status(job["id"], JobStatus.TRANSCRIBING.value, progress=30)
        vad_segments, detected_language = await asyncio.to_thread(transcribe, wav_path, cfg.transcription)
        wav_path.unlink(missing_ok=True)  # clean up early — punctuate only needs text
        effective_language = (
            cfg.transcription.language if cfg.transcription.language != "auto" else (detected_language or "en")
        )
        sentence_segments = await asyncio.to_thread(punctuate, vad_segments, effective_language)

        if await self._abort_if_cancelled(job):
            return None

        if not sentence_segments:
            # Music-only / speech-less video: fail loud here instead of an
            # opaque IndexError in digest (sections would be empty). Checked
            # after the cancel gate so a user cancel wins over the failure.
            raise PipelineError("no speech detected in audio")

        return detected_language, effective_language, sentence_segments

    async def _stage_process(
        self,
        job: dict,
        sentence_segments: list[WhisperSegment],
        source_id: str,
        video_meta: VideoMeta,
        list_id: str,
        cfg: BibilabConfig,
        effective_language: str,
    ) -> tuple[DigestResult, list[Section], list[SectionDigest]] | None:
        """Stage 4: Derive sections, chunk per-section, run digest + embed in parallel.

        Returns (extraction, sections, section_digests). The 1-section case
        produces a 1-element section_digests mirroring the DigestResult.

        Pipeline order: `segments → sections → chunks` (section-first, chunk-within).
        Chunks are produced from per-section slices, so a chunk can never cross a
        section boundary — the `chunk → section → source` nesting is physical, not
        by convention. Re-stamping in `chunk_by_sections` keeps the chunks' seg
        indices and sequence_index source-global so the rest of the system is
        unaware sections exist (the per-chunk section FK is a future
        optimization; today the chunk→section containment is a structural
        property of chunk_by_sections, not a stored relationship).
        """
        await update_job_status(job["id"], JobStatus.PROCESSING.value, progress=40)
        sections = derive_sections(sentence_segments)
        chunks = chunk_by_sections(sentence_segments, sections, language=effective_language)
        section_texts_list = section_texts(sentence_segments, sections)

        meta_raw = parse_job_meta(job)
        ui_lang = meta_raw.get("ui_lang")

        async def _digest() -> tuple[DigestResult, list[SectionDigest]]:
            return await self._call_digest_sections(section_texts_list, video_meta, cfg, ui_lang)

        async def _embed() -> None:
            await asyncio.to_thread(embed_chunks, chunks, source_id, video_meta, list_id)

        gather_results = await asyncio.gather(_digest(), _embed(), return_exceptions=True)
        digest_raw, embed_raw = gather_results
        _reraise_gathered_failures(digest_raw, embed_raw)
        extraction, section_digests = digest_raw

        if await self._abort_if_cancelled(job):
            return None

        return extraction, sections, section_digests

    async def _stage_persist(
        self,
        job_id: str,
        source_id: str,
        video_id: str,
        video_meta: VideoMeta,
        list_id: str,
        extraction: DigestResult,
        sections: list[Section],
        section_digests: list[SectionDigest],
        detected_language: str,
        cfg: BibilabConfig,
        sentence_segments: list[WhisperSegment],
    ) -> None:
        """Stage 5: Persist source row + transcript segments + section rows
        atomically, then cleanup. All three land in one transaction (or none);
        a partial write rolls back so re-ingest never leaves orphan rows."""
        await write_source_with_segments(
            segments=sentence_segments,
            sections=sections,
            section_digests=section_digests,
            source_id=source_id,
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=video_meta.title,
            cover_url=video_meta.cover_url,
            source_url=video_meta.source_url,
            duration_seconds=video_meta.duration_seconds,
            uploader=video_meta.uploader,
            language=detected_language,
            whisper_model=cfg.transcription.model,
            ai_model=cfg.ai.model,
            settings_snapshot=cfg.model_dump(),
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
        )

        # Cleanup downloads (off the loop — unlinking multi-GB files blocks on WSL)
        await asyncio.to_thread(purge_download_files, video_id)
