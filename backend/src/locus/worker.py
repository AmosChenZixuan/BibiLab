"""Background worker loop. Picks up queued jobs and drives them through the pipeline."""

import asyncio
import logging

from locus.db import get_pending_jobs, reset_stuck_jobs, update_job_status

logger = logging.getLogger(__name__)

# Set by DELETE /jobs/{id} to signal a running job should stop
_cancelled: set[str] = set()


class WorkerLoop:
    def __init__(self, concurrency: int = 1) -> None:
        self._concurrency = concurrency
        self._task: asyncio.Task | None = None
        self._running = False

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
        _cancelled.add(job_id)

    async def _loop(self) -> None:
        while self._running:
            pending = await get_pending_jobs()
            active = [j for j in pending if dict(j)["status"] != "queued"]

            if len(active) < self._concurrency:
                queued = [j for j in pending if dict(j)["status"] == "queued"]
                slots = self._concurrency - len(active)
                for job in queued[:slots]:
                    asyncio.create_task(self._run_job(dict(job)))

            await asyncio.sleep(2)

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        if job_id in _cancelled:
            _cancelled.discard(job_id)
            await update_job_status(job_id, "failed", error="Cancelled")
            return

        try:
            # Stub pipeline — replaced in Phase 4
            await update_job_status(job_id, "downloading", progress=10)
            await asyncio.sleep(0)  # yield

            if job_id in _cancelled:
                _cancelled.discard(job_id)
                await update_job_status(job_id, "failed", error="Cancelled")
                return

            await update_job_status(job_id, "done", progress=100)
            logger.info("Job %s completed (stub)", job_id)

        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await update_job_status(job_id, "failed", error=str(exc))
