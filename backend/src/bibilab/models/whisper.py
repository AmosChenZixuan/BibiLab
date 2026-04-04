from pydantic import BaseModel


class WhisperModelInfo(BaseModel):
    name: str
    installed: bool
    path: str | None
    selected: bool


class WhisperModelDownloadRequest(BaseModel):
    model_size: str


class WhisperModelDownloadResponse(BaseModel):
    name: str
    path: str
    selected: bool
