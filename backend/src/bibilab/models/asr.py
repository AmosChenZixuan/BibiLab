"""Pydantic models for unified ASR model endpoints."""

from typing import Literal

from pydantic import BaseModel, model_validator

from bibilab.config import SUPPORTED_MODELS, AsrModelKind


class AsrModelInfo(BaseModel):
    name: str
    engine: AsrModelKind
    installed: bool
    path: str | None
    selected: bool


class AsrModelDownloadRequest(BaseModel):
    engine: AsrModelKind
    model_size: str

    @model_validator(mode="after")
    def _check_supported(self) -> "AsrModelDownloadRequest":
        if self.model_size not in SUPPORTED_MODELS[self.engine]:
            raise ValueError(f"Unsupported model {self.model_size!r} for engine {self.engine!r}")
        return self


class AsrModelDownloadResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    engine: AsrModelKind
    model_size: str
