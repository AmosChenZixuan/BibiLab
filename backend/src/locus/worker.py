"""Background worker loop."""

import asyncio
import json
import logging
from pathlib import Path

from locus.adapters.base import AuthRequiredError, VideoMeta
from locus.adapters.bilibili import BilibiliAdapter
from locus.cleanup import cleanup_job_artifacts
from locus.config import load_config, locus_home
from locus.db import (
    delete_job,
    get_list,
    get_pending_jobs,
    reset_stuck_jobs,
    update_job_status,
    write_source,
)
from locus.pipeline.audio import PipelineError, extract_audio
from locus.pipeline.chunk import chunk_segments
from locus.pipeline.embed import embed_chunks, is_embedding_model_downloaded
from locus.pipeline.extract import ExtractionResult, extract_knowledge
from locus.pipeline.notes import write_video_note
from locus.pipeline.transcribe import transcribe, write_transcript
from locus.whisper_models import download_whisper_model

logger = logging.getLogger(__name__)


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
        self._task = asyncio.create_task(self._loop(), name="locus-worker")
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
                    j for j in pending if j["status"] == "queued" and j["id"] not in self._in_flight
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

            await self._pipeline(job)

        except AuthRequiredError as exc:
            logger.warning("Job %s needs auth (%s)", job_id, exc.resource_type)
            await update_job_status(job_id, "needs_auth", error=exc.resource_type)
        except PipelineError as exc:
            logger.error("Job %s pipeline error: %s", job_id, exc)
            await update_job_status(job_id, "failed", error=str(exc))
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await update_job_status(job_id, "failed", error=str(exc))
        finally:
            self._in_flight.discard(job_id)
            self._cancelled.discard(job_id)

    async def _download_model_job(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = _parse_job_meta(job)
        model_family = meta_raw.get("model_family", "")
        model_size = meta_raw.get("model_size", "")

        await update_job_status(job_id, "downloading", progress=10)
        if model_family != "whisper":
            raise PipelineError(f"Unsupported model family {model_family!r}")

        await asyncio.to_thread(download_whisper_model, model_size)
        await update_job_status(job_id, "done", progress=100)
        logger.info("Model download job %s completed for %s:%s", job_id, model_family, model_size)

    async def _pipeline(self, job: dict) -> None:
        job_id = job["id"]
        meta_raw = _parse_job_meta(job)
        cfg = load_config()

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

        # 1. Download
        await update_job_status(job_id, "downloading", progress=10)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return

        adapter = BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)
        video_path: Path = await asyncio.to_thread(
            adapter.download, video_id, video_meta.source_url
        )

        # 2. Audio extraction
        await update_job_status(job_id, "transcribing", progress=25)
        wav_path: Path = await asyncio.to_thread(extract_audio, video_path)

        # 3. Transcription
        segments = await asyncio.to_thread(transcribe, wav_path, cfg.transcription)
        transcript_path = write_transcript(segments, video_id)
        wav_path.unlink(missing_ok=True)

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return

        # 4. Chunking + extraction
        await update_job_status(job_id, "extracting", progress=40)
        chunks = chunk_segments(segments)
        transcript_text = transcript_path.read_text(encoding="utf-8")
        extraction: ExtractionResult = await asyncio.to_thread(
            extract_knowledge, transcript_text, video_meta, cfg.ai
        )

        if extraction.title:
            video_meta = VideoMeta(
                video_id=video_meta.video_id,
                title=extraction.title,
                platform=video_meta.platform,
                source_url=video_meta.source_url,
                cover_url=video_meta.cover_url,
                duration_seconds=video_meta.duration_seconds,
                uploader=video_meta.uploader,
            )

        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return

        # 5. Write note + embed
        await update_job_status(job_id, "writing", progress=70)
        note_path = await asyncio.to_thread(write_video_note, video_meta, extraction, list_id)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return
        if not is_embedding_model_downloaded():
            logger.info("Job %s: embedding model not found - downloading (~50 MB).", job_id)
        await asyncio.to_thread(embed_chunks, chunks, video_meta, list_id, cfg)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            await asyncio.to_thread(cleanup_job_artifacts, job)
            await delete_job(job_id)
            return

        # 6. Persist processed source metadata
        await write_source(
            video_id=video_id,
            platform=video_meta.platform,
            list_id=list_id,
            title=extraction.title or video_id,
            summary=extraction.summary,
            note_path=str(note_path),
            transcript_path=str(transcript_path),
            whisper_model=cfg.transcription.model_size,
            ai_model=cfg.ai.model,
            vision_enabled=cfg.vision.enabled,
            settings_snapshot=cfg.model_dump(),
            cover_url=video_meta.cover_url,
        )

        # 7. Cleanup downloads
        for path in (locus_home() / "downloads").glob(f"{video_id}.*"):
            path.unlink(missing_ok=True)

        await update_job_status(job_id, "done", progress=100)
        logger.info("Job %s completed for video %s", job_id, video_id)
