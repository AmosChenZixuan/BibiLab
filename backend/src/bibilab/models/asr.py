"""Pydantic models for ASR endpoints."""

from typing import Literal

from pydantic import BaseModel

from bibilab.config import AsrModelKind


class AsrModelInfo(BaseModel):
    name: str
    kind: AsrModelKind
    installed: bool
    path: str | None
    selected: bool
    size_mb: int


class AsrModelDownloadRequest(BaseModel):
    model_name: str


class AsrModelDownloadResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    model_name: str
