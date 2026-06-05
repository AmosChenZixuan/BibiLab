"""Background worker loop."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from bibilab.adapters.base import AuthRequiredError, VideoMeta
from bibilab.cleanup import cleanup_job_artifacts
from bibilab.config import BibilabConfig, bibilab_home, load_config
from bibilab.db import (
    create_artifact,
    delete_job,
    get_list,
    get_pending_jobs,
    get_source,
    parse_job_meta,
    reset_stuck_jobs,
    update_job_meta,
    update_job_status,
    update_source_digest,
    write_source_with_segments,
)
from bibilab.model_registry import ensure
from bibilab.models.jobs import JobStatus
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _call_llm,
    _lang_output_directive,
    _parse_llm_json_response,
    _resolved_lang,
)
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import chunk_segments
from bibilab.pipeline.digest import DigestResult, digest
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.punctuate import punctuate
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    format_turns,
    load_transcript_text,
    transcribe,
)

logger = logging.getLogger(__name__)

# Sized for thinking-capable models with multi-source artifact output (multi-page markdown
# report). Thinking + long-form output can easily exceed 16K for complex artifacts.
ARTIFACT_MAX_TOKENS = 32768


class ArtifactResult(BaseModel):
    """Result from LLM artifact generation."""

    name: str
    content: str


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

    def _get_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        cfg = self._get_config()
        from bibilab.adapters.bilibili import BilibiliAdapter

        return BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)

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

            if job["type"] == "artifact":
                await self._run_artifact_job(job)
                return

            if job["type"] == "model_download":
                await self._download_model_job(job)
                return

            if job["type"] == "digest":
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
            if job.get("type") in (None, "ingest"):
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
        error_message: str | None = None
        artifact_result: ArtifactResult | None = None

        try:
            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

            # Load transcripts for all source_ids
            transcripts: list[str] = []
            for source_id in source_ids:
                source = await get_source(source_id)
                if source is None:
                    raise PipelineError(f"Source {source_id!r} not found")
                transcript_text = await load_transcript_text(source_id, include_time=False)
                if not transcript_text:
                    raise PipelineError(f"Source {source_id!r} has no transcript")
                transcripts.append(f"=== Source: {source['title']} ===\n{transcript_text}")

            combined_transcript = "\n\n".join(transcripts)

            # Check for cancellation before LLM call
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, JobStatus.PROCESSING.value, progress=30)

            # Call LLM to generate artifact
            artifact_result = await self._generate_artifact(
                prompt, artifact_type, combined_transcript, cfg, meta_raw.get("ui_lang")
            )

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
            content_path.write_text(artifact_result.content, encoding="utf-8")

            # Create artifact record with success status
            await create_artifact(
                artifact_id=artifact_id,
                list_id=list_id,
                name=artifact_result.name,
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

    async def _call_digest(
        self,
        transcript_text: str,
        video_meta: VideoMeta,
        cfg: BibilabConfig,
        ui_lang: str | None = None,
    ) -> DigestResult:
        """Run the digest LLM call. Does not catch exceptions — callers handle errors."""
        return await asyncio.to_thread(
            digest,
            transcript_text,
            video_meta,
            cfg.ai,
            cfg.ai.output_language,
            ui_lang,
            llm_timeout=cfg.transcription.llm_timeout,
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

        transcript_text = await load_transcript_text(source_id)
        if not transcript_text:
            logger.warning("Digest job %s: source %s has no transcript", job_id, source_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error="Source has no transcript")
            return

        video_meta = VideoMeta.from_source(source)

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=10)

        try:
            extraction = await self._call_digest(transcript_text, video_meta, cfg, meta_raw.get("ui_lang"))
        except Exception as exc:
            logger.exception("Digest job %s failed", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
            return

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=80)

        await update_source_digest(
            source_id,
            extraction.summary,
            extraction.keywords,
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
            bump_processed_at=False,
        )

        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Digest job %s completed for source %s", job_id, source_id)

    async def _generate_artifact(
        self,
        prompt: str,
        artifact_type: str,
        transcript_text: str,
        cfg: BibilabConfig,
        ui_lang: str | None = None,
    ) -> ArtifactResult:
        """Call LLM to generate artifact content. Returns title and content."""
        lang = _resolved_lang(cfg.ai.output_language, ui_lang)
        lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])

        llm_prompt = f"""{lang_instruction}

{prompt}

Based on the following transcripts, generate the requested artifact content.

Transcript:
{transcript_text}

{lang_instruction}
Respond ONLY with valid JSON matching this schema:
{{
  "name": "string (a short title for this artifact)",
  "content": "string (the main artifact content in markdown format)"
}}
{_lang_output_directive(lang)}"""

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                raw = await asyncio.to_thread(
                    _call_llm,
                    llm_prompt,
                    cfg.ai,
                    llm_timeout=cfg.transcription.llm_timeout,
                    llm_max_tokens=ARTIFACT_MAX_TOKENS,
                )
                return _parse_llm_json_response(raw, ArtifactResult)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "LLM artifact generation failed (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    continue

        error_msg = f"LLM artifact generation exhausted all retries: {last_exc}"
        logger.error(error_msg)
        # On failure, create artifact with error status
        raise PipelineError(error_msg)

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
            platform=meta_raw.get("platform", ""),
            source_url=meta_raw.get("source_url", ""),
            cover_url=meta_raw.get("cover_url", ""),
            duration_seconds=meta_raw.get("duration_seconds", 0),
            uploader=meta_raw.get("uploader", ""),
        )

        # Stage 1: Download
        try:
            video_path = await self._stage_download(job_id, video_meta, source_id)
        except Exception as exc:
            raise PipelineError(f"[downloading] {exc}") from exc
        if video_path is None:
            return  # cancelled

        # Stage 2: Audio extraction
        try:
            wav_path = await self._stage_extract_audio(job_id, video_path)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if wav_path is None:
            return  # cancelled

        # Stage 3: Transcription
        try:
            result = await self._stage_transcribe(job_id, wav_path, source_id, cfg)
        except Exception as exc:
            raise PipelineError(f"[transcribing] {exc}") from exc
        if result is None:
            return  # cancelled
        detected_language, effective_language, sentence_segments = result

        # Stage 4: Chunk, digest + embed in parallel
        try:
            extraction = await self._stage_process(
                job_id, job, sentence_segments, source_id, video_meta, list_id, cfg, effective_language
            )
        except Exception as exc:
            raise PipelineError(f"[processing] {exc}") from exc
        if extraction is None:
            return  # cancelled

        # Stage 5: Persist + cleanup
        try:
            await self._stage_persist(
                job_id,
                source_id,
                video_id,
                video_meta,
                list_id,
                extraction,
                detected_language,
                cfg,
                sentence_segments,
            )
        except Exception as exc:
            raise PipelineError(f"[processing] {exc}") from exc
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Job %s completed for video %s", job_id, video_id)

    async def _stage_download(
        self,
        job_id: str,
        video_meta: VideoMeta,
        source_id: str,
    ) -> Path | None:
        """Stage 1: Download video file and cover image."""
        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        video_path: Path = await asyncio.to_thread(
            self._get_adapter().download,
            video_meta.video_id,
            video_meta.source_url,
        )

        # Download cover
        covers_dir = self._bibilab_home / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        cover_dest = covers_dir / f"{source_id}.jpg"
        await asyncio.to_thread(_download_cover, video_meta.cover_url, cover_dest)

        return video_path

    async def _stage_extract_audio(
        self,
        job_id: str,
        video_path: Path,
    ) -> Path | None:
        """Stage 2: Extract audio from downloaded video."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=25)
        try:
            wav_path = await asyncio.to_thread(extract_audio, video_path)
        except Exception:
            # video_path exists but audio extraction failed - video will be cleaned up
            # by cancel_or_delete_job using the full job dict from DB
            raise

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        return wav_path

    async def _stage_transcribe(
        self,
        job_id: str,
        wav_path: Path,
        source_id: str,
        cfg: BibilabConfig,
    ) -> tuple | None:
        """Stage 3: Transcribe audio, punctuate (zh-gated) into sentence segments."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=30)
        vad_segments, detected_language = await asyncio.to_thread(transcribe, wav_path, cfg.transcription)
        wav_path.unlink(missing_ok=True)  # clean up early — punctuate only needs text
        effective_language = (
            cfg.transcription.language if cfg.transcription.language != "auto" else (detected_language or "en")
        )
        sentence_segments = await asyncio.to_thread(punctuate, vad_segments, effective_language)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        return detected_language, effective_language, sentence_segments

    async def _stage_process(
        self,
        job_id: str,
        job: dict,
        sentence_segments: list[WhisperSegment],
        source_id: str,
        video_meta: VideoMeta,
        list_id: str,
        cfg: BibilabConfig,
        effective_language: str,
    ) -> DigestResult | None:
        """Stage 4: Chunk sentence segments, run digest + embed in parallel."""
        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=40)
        chunks = chunk_segments(
            sentence_segments,
            language=effective_language,
            pause_threshold_seconds=cfg.rag.chunk_pause_threshold,
        )

        meta_raw = parse_job_meta(job)
        transcript_text = format_turns(sentence_segments, include_time=False)

        async def _digest() -> DigestResult:
            return await self._call_digest(transcript_text, video_meta, cfg, meta_raw.get("ui_lang"))

        async def _embed() -> None:
            await asyncio.to_thread(embed_chunks, chunks, source_id, video_meta, list_id, cfg)

        extraction: DigestResult
        extraction, _ = await asyncio.gather(_digest(), _embed())

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return None

        return extraction

    async def _stage_persist(
        self,
        job_id: str,
        source_id: str,
        video_id: str,
        video_meta: VideoMeta,
        list_id: str,
        extraction: DigestResult,
        detected_language: str,
        cfg: BibilabConfig,
        sentence_segments: list[WhisperSegment],
    ) -> None:
        """Stage 5: Persist source row + transcript segments atomically, then cleanup."""
        # One transaction: a source row never commits without its segments (no
        # orphaned-source state, no compensating delete). See write_source_with_segments.
        await write_source_with_segments(
            segments=sentence_segments,
            source_id=source_id,
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=video_meta.title,
            summary=extraction.summary,
            keywords=extraction.keywords,
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

        # Cleanup downloads
        for path in (self._bibilab_home / "downloads").glob(f"{video_id}.*"):
            path.unlink(missing_ok=True)
