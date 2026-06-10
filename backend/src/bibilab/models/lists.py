from pydantic import BaseModel


class PlatformResource(BaseModel):
    """Shared base for source responses with common platform resource fields."""

    id: str
    title: str
    source_url: str
    cover_url: str | None
    processed_at: str  # ISO 8601 string from DB


class ListCreateRequest(BaseModel):
    name: str


class ListUpdateRequest(BaseModel):
    name: str | None = None
    thumbnail_source_id: str | None = None


class ListResponse(BaseModel):
    id: str
    name: str
    created_at: str
    thumbnail_source_id: str | None
    thumbnail_url: str | None
    source_count: int
    updated_at: str


class SourceResponse(PlatformResource):
    """Source response inheriting shared platform resource fields."""

    video_id: str
    platform: str
    duration_seconds: int
    uploader: str
    language: str | None
