import json
import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from bibilab.cleanup import cleanup_job_artifacts
from bibilab.db import delete_job, get_job, list_jobs
from bibilab.models.jobs import TERMINAL_STATUSES, JobResponse, JobStatus

router = APIRouter()
logger = logging.getLogger(__name__)


def _row_to_response(row: sqlite3.Row) -> JobResponse:
    d = dict(row)
    return JobResponse(
        id=d["id"],
        type=d["type"],
        status=JobStatus(d["status"]),
        progress=d["progress"] or 0,
        error=d["error"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        meta=json.loads(d["meta"] or "{}"),
    )


@router.get("/jobs")
async def get_jobs() -> list[JobResponse]:
    rows = await list_jobs()
    return [_row_to_response(r) for r in rows]


@router.get("/jobs/{job_id}")
async def get_job_by_id(job_id: str) -> JobResponse:
    """Single-job metadata. Unused by the web SPA (JobActivityProvider polls the
    list endpoint); kept as REST surface for ad-hoc inspection / debugging."""
    row = await get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_response(row)


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_or_delete_job(job_id: str, request: Request) -> None:
    row = await get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(row)
    status = JobStatus(job["status"])

    if status not in TERMINAL_STATUSES and status != JobStatus.QUEUED:
        worker = request.app.state.worker
        if worker:
            worker.cancel_job(job_id)

    try:
        cleanup_job_artifacts(job)
    except Exception:
        logger.exception("Failed to clean up artifacts for job %s", job_id)

    await delete_job(job_id)
