import json

from fastapi import APIRouter, HTTPException, Request

from locus.db import delete_job, get_job, list_jobs
from locus.models.jobs import JobResponse, JobStatus, TERMINAL_STATUSES

router = APIRouter()


def _row_to_response(row) -> JobResponse:
    d = dict(row)
    return JobResponse(
        id=d["id"],
        type=d["type"],
        source_url=d["source_url"],
        platform=d["platform"],
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
    row = await get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_response(row)


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_or_delete_job(job_id: str, request: Request) -> None:
    row = await get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = JobStatus(dict(row)["status"])

    if status == JobStatus.QUEUED or status in TERMINAL_STATUSES:
        await delete_job(job_id)
    else:
        # In-progress: signal worker to stop; it will set status to failed
        worker = request.app.state.worker
        if worker:
            worker.cancel_job(job_id)
