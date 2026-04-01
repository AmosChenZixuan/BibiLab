from pydantic import BaseModel


class ListCreateRequest(BaseModel):
    name: str


class ListUpdateRequest(BaseModel):
    name: str


class ListResponse(BaseModel):
    id: str
    name: str
    created_at: str


class SourceResponse(BaseModel):
    video_id: str
    platform: str
    title: str
    note_path: str
    processed_at: str


class OverviewResponse(BaseModel):
    content: str
    filename: str
