from pydantic import BaseModel

from bibilab.models._enums import VideoStatus


class IngestVideoIn(BaseModel):
    video_id: str
    title: str
    cover_url: str
    duration_seconds: int
    uploader: str
    platform: str
    source_url: str


class IngestUrlRequest(BaseModel):
    list_id: str
    videos: list[IngestVideoIn]


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


class VideoMetadata(BaseModel):
    title: str
    cover_url: str
    duration_seconds: int
    uploader: str
    source_url: str
    part_label: str | None = None


class VideoMetadataRequest(BaseModel):
    video_ids: list[str]
    # No default: the client must say which platform's adapter to use — a
    # silent bilibili assumption misroutes every other platform.
    platform: str


class VideoMetadataMapResponse(BaseModel):
    videos: dict[str, VideoMetadata]
    expanded: dict[str, list[str]] = {}
