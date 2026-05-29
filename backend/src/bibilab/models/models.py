"""Pydantic models for model registry endpoints."""

from typing import Literal

from pydantic import BaseModel

ModelStatus = Literal["present", "missing"]


class ModelInfo(BaseModel):
    id: str
    display_name: str
    kind: str
    size_mb: int
    status: ModelStatus
    required_by_config: bool
    path: str | None = None


class ModelDownloadResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    spec_id: str


class SyncResponse(BaseModel):
    job_ids: list[str]
    synced: list[str]
    skipped: list[str]
