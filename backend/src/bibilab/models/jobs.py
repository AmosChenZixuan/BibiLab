from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    WRITING = "writing"
    DONE = "done"
    FAILED = "failed"
    NEEDS_AUTH = "needs_auth"


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED, JobStatus.NEEDS_AUTH}


class JobResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: int
    error: str | None
    created_at: str
    updated_at: str
    meta: dict[str, Any]
