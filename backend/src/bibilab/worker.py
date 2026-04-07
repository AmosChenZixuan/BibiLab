"""Background worker loop."""

import asyncio
import json
import logging
import uuid
from pathlib import Path

import httpx

from bibilab.adapters.base import AuthRequiredError, VideoMeta
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.cleanup import cleanup_job_artifacts
from bibilab.config import bibilab_home, load_config
from bibilab.db import (
    delete_job,
    get_list,
    get_pending_jobs,
    get_source,
    reset_stuck_jobs,
    update_job_status,
    update_source_digest,
    write_source,
)
from bibilab.models.jobs import JobStatus
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import chunk_segments
from bibilab.pipeline.digest import DigestResult, digest
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.transcribe import transcribe, write_transcript
from bibilab.whisper_models import download_whisper_model

logger = logging.getLogger(__name__)

# Lazy singleton for BilibiliAdapter (avoids per-request instantiation)
_bilibili_adapter: BilibiliAdapter | None = None


def _get_bilibili_adapter() -> BilibiliAdapter:
    global _bilibili_adapter
    if _bilibili_adapter is None:
        _bilibili_adapter = BilibiliAdapter(cookie=load_config().accounts.bilibili.cookie)
    return _bilibili_adapter


def _download_cover(cover_url: str, dest: Path) -> bool:
    """Download cover image from URL to dest path. Returns True on success."""
    try:
        resp = httpx.get(cover_url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception:
        return False


def _parse_job_meta(job: dict) -> dict:
    meta = job.get("meta", {})
    if isinstance(meta, str):
        return json.loads(meta or "{}")
    return meta


class WorkerLoop:
    def __init__(self, concurrency: int = 1) -> None:
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

            if job["type"] == "model_download":
                await self._download_model_job(job)
                return

            meta_raw = _parse_job_meta(job)
            if meta_raw.get("stages") == ["digest"]:
                await self._pipeline_digest_only(job)
                return

            await self._pipeline(job)

        except AuthRequiredError as exc:
            logger.warning("Job %s needs auth (%s)", job_id, exc.resource_type)
            await update_job_status(job_id, JobStatus.NEEDS_AUTH.value, error=exc.resource_type)
        except PipelineError as exc:
            logger.error("Job %s pipeline error: %s", job_id, exc)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await update_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
        finally:
            self._in_flight.discard(job_id)
            self._cancelled.discard(job_id)

    async def _download_model_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = _parse_job_meta(job)
        model_family = meta_raw.get("model_family", "")
        model_size = meta_raw.get("model_size", "")

        await update_job_status(job_id, JobStatus.DOWNLOADING.value, progress=10)
        if model_family != "whisper":
            raise PipelineError(f"Unsupported model family {model_family!r}")

        await asyncio.to_thread(download_whisper_model, model_size)
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)
        logger.info("Model download job %s completed for %s:%s", job_id, model_family, model_size)

    # -------------------------------------------------------------------------
    # Pipeline stages (digest-only)
    # -------------------------------------------------------------------------

    async def _pipeline_digest_only(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = _parse_job_meta(job)
        source_id = meta_raw["source_id"]
        cfg = load_config()

        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=40)
        source = await get_source(source_id)
        if source is None:
            raise PipelineError(f"Source {source_id!r} not found")
        if source["transcript_path"] is None:
            raise PipelineError(f"Source {source_id!r} has no transcript")

        transcript_path = bibilab_home() / source["transcript_path"]
        transcript_text = transcript_path.read_text(encoding="utf-8")

        video_meta = VideoMeta.from_source(source)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await delete_job(job_id)
            return

        extraction = await asyncio.to_thread(
            digest,
            transcript_text,
            video_meta,
            cfg.ai,
            cfg.ai.output_language,
            meta_raw.get("ui_lang"),
        )
        await update_source_digest(source_id, extraction.summary, extraction.keywords)
        await update_job_status(job_id, JobStatus.DONE.value, progress=100)

    # -------------------------------------------------------------------------
    # Pipeline stages (full ingest)
    # -------------------------------------------------------------------------

    async def _pipeline(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = _parse_job_meta(job)
        cfg = load_config()

        source_id: str = meta_raw.get("source_id") or str(uuid.uuid4())
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
        video_path = await self._stage_download(job_id, video_meta, source_id)
        if video_path is None:
            return  # cancelled

        # Stage 2: Audio extraction
        wav_path = await self._stage_extract_audio(job_id, video_path)
        if wav_path is None:
            return  # cancelled

        # Stage 3: Transcription
        result = await self._stage_transcribe(job_id, wav_path, source_id, cfg)
        if result is None:
            return  # cancelled
        segments, detected_language, transcript_path = result

        # Stage 4: Chunk, digest + embed in parallel
        extraction = await self._stage_process(job_id, job, segments, video_meta, list_id, cfg, transcript_path)
        if extraction is None:
            return  # cancelled

        # Stage 5: Persist + cleanup
        await self._stage_persist(
            job_id,
            source_id,
            video_id,
            video_meta,
            list_id,
            extraction,
            detected_language,
            cfg,
            transcript_path,
        )
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

        adapter = _get_bilibili_adapter()
        video_path: Path = await asyncio.to_thread(
            adapter.download,
            video_meta.video_id,
            video_meta.source_url,
        )

        # Download cover
        covers_dir = bibilab_home() / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        cover_dest = covers_dir / f"{source_id}.jpg"
        try:
            await asyncio.to_thread(_download_cover, video_meta.cover_url, cover_dest)
        except Exception:
            logger.warning("Job %s: failed to download cover, continuing without local cover", job_id)

        return video_path

    async def _stage_extract_audio(
        self,
        job_id: str,
        video_path: Path,
    ) -> Path | None:
        """Stage 2: Extract audio from downloaded video."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=25)
        return await asyncio.to_thread(extract_audio, video_path)

    async def _stage_transcribe(
        self,
        job_id: str,
        wav_path: Path,
        source_id: str,
        cfg,
    ) -> tuple | None:
        """Stage 3: Transcribe audio and write transcript file."""
        await update_job_status(job_id, JobStatus.TRANSCRIBING.value, progress=30)
        segments, detected_language = await asyncio.to_thread(transcribe, wav_path, cfg.transcription)
        transcript_path = write_transcript(segments, source_id)
        wav_path.unlink(missing_ok=True)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, {"id": job_id})
            await delete_job(job_id)
            return None

        return segments, detected_language, transcript_path

    async def _stage_process(
        self,
        job_id: str,
        job: dict,
        segments,
        video_meta: VideoMeta,
        list_id: str,
        cfg,
        transcript_path: Path,
    ) -> DigestResult | None:
        """Stage 4: Chunk segments, run digest + embed in parallel."""
        await update_job_status(job_id, JobStatus.PROCESSING.value, progress=40)
        chunks = chunk_segments(segments)

        meta_raw = _parse_job_meta(job)
        transcript_text = transcript_path.read_text(encoding="utf-8")

        async def _digest():
            return await asyncio.to_thread(
                digest,
                transcript_text,
                video_meta,
                cfg.ai,
                cfg.ai.output_language,
                meta_raw.get("ui_lang"),
            )

        async def _embed():
            await asyncio.to_thread(embed_chunks, chunks, video_meta, list_id, cfg)

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
        cfg,
        transcript_path: Path,
    ) -> None:
        """Stage 5: Persist processed source metadata and cleanup downloads."""
        await write_source(
            source_id=source_id,
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=video_meta.title,
            summary=extraction.summary,
            keywords=extraction.keywords,
            cover_url=video_meta.cover_url,
            transcript_path=str(transcript_path.relative_to(bibilab_home())),
            source_url=video_meta.source_url,
            duration_seconds=video_meta.duration_seconds,
            uploader=video_meta.uploader,
            language=detected_language,
            whisper_model=cfg.transcription.model_size,
            ai_model=cfg.ai.model,
            vision_enabled=cfg.vision.enabled,
            settings_snapshot=cfg.model_dump(),
        )

        # Cleanup downloads
        for path in (bibilab_home() / "downloads").glob(f"{video_id}.*"):
            path.unlink(missing_ok=True)
