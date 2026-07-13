from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class JobType(StrEnum):
    INGEST = "ingest"
    DIGEST = "digest"
    ARTIFACT = "artifact"
    MODEL_DOWNLOAD = "model_download"


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    NEEDS_AUTH = "needs_auth"


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED, JobStatus.NEEDS_AUTH}

ACTIVE_JOB_STATUSES: tuple[JobStatus, ...] = (
    JobStatus.QUEUED,
    JobStatus.DOWNLOADING,
    JobStatus.TRANSCRIBING,
    JobStatus.PROCESSING,
)


class JobResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: int
    error: str | None
    created_at: str
    updated_at: str
    meta: dict[str, Any]
