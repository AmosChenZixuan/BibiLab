from pydantic import BaseModel


class ListCreateRequest(BaseModel):
    name: str


class ListResponse(BaseModel):
    id: str
    name: str
    created_at: str


class ListNoteResponse(BaseModel):
    video_id: str
    note_path: str | None
    processed_at: str
    platform: str
