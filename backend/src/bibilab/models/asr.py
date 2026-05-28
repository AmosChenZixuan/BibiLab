"""Pydantic models for unified ASR model endpoints."""

from pydantic import BaseModel


class AsrModelInfo(BaseModel):
    name: str
    engine: str  # "whisper" | "sensevoice" | "diarization"
    installed: bool
    path: str | None
    selected: bool


class AsrModelDownloadRequest(BaseModel):
    engine: str
    model_size: str


class AsrModelDownloadResponse(BaseModel):
    job_id: str
    status: str
    engine: str
    model_size: str
