from typing import Any

from pydantic import BaseModel


class SourceContentResponse(BaseModel):
    id: str
    video_id: str
    platform: str
    title: str
    source_url: str
    duration_seconds: int
    uploader: str
    language: str | None
    processed_at: str
    summary: str
    keywords: list[str]
    cover_url: str | None
    transcript: str
    settings_snapshot: dict[str, Any]
