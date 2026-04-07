from pydantic import BaseModel


class WhisperModelInfo(BaseModel):
    name: str
    installed: bool
    path: str | None
    selected: bool


class WhisperModelDownloadRequest(BaseModel):
    model_size: str


class WhisperModelDownloadResponse(BaseModel):
    """Response model for Whisper model download endpoint."""

    job_id: str
    status: str
    model_family: str
    model_size: str
