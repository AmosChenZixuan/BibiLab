from enum import StrEnum
from typing import Any, TypedDict

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


class JobMeta(TypedDict, total=False):
    """Typed schema for job metadata.

    The 'meta' field in jobs stores operation-specific data.
    This TypedDict captures common fields but remains flexible
    since different job types (ingest, model_download, digest-only)
    store different data.
    """

    video_id: str
    list_id: str
    title: str
    cover_url: str
    duration_seconds: int
    uploader: str
    source_url: str
    platform: str
    ui_lang: str
    source_id: str
    stages: list[str]
    model_family: str
    model_size: str


class JobResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: int
    error: str | None
    created_at: str
    updated_at: str
    meta: dict[str, Any]
