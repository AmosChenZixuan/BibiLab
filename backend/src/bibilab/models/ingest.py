from pydantic import BaseModel

from bibilab.models._enums import VideoStatus


class IngestUrlRequest(BaseModel):
    list_id: str
    url: str


class IngestUrlResponse(BaseModel):
    queued: list[str]
    skipped: list[str]


class IngestPreviewRequest(BaseModel):
    list_id: str
    url: str


class PreviewVideo(BaseModel):
    video_id: str
    title: str
    cover_url: str
    duration_seconds: int
    uploader: str
    platform: str
    source_url: str
    part_label: str | None = None
    status: VideoStatus


class IngestPreviewResponse(BaseModel):
    videos: list[PreviewVideo]
