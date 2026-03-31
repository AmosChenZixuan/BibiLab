"""Background worker loop. Picks up queued jobs and drives them through the pipeline."""

import asyncio
import logging

from locus.db import delete_job, get_pending_jobs, reset_stuck_jobs, update_job_status

logger = logging.getLogger(__name__)


class WorkerLoop:
    def __init__(self, concurrency: int = 1) -> None:
        self._concurrency = concurrency
        self._task: asyncio.Task | None = None
        self._running = False
        self._cancelled: set[str] = set()
        self._in_flight: set[str] = set()  # job IDs currently being processed

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
                    j
                    for j in pending
                    if j["status"] == "queued" and j["id"] not in self._in_flight
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

            # Stub pipeline — replaced in Phase 4
            await update_job_status(job_id, "downloading", progress=10)
            await asyncio.sleep(0)  # yield

            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                await delete_job(job_id)
                return

            await update_job_status(job_id, "done", progress=100)
            logger.info("Job %s completed (stub)", job_id)

        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            await update_job_status(job_id, "failed", error=str(exc))
        finally:
            self._in_flight.discard(job_id)
            self._cancelled.discard(job_id)  # clean up even if job was never dispatched
