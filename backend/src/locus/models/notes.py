from pydantic import BaseModel


class NoteContentResponse(BaseModel):
    video_id: str
    title: str
    markdown: str


class NoteTranscriptResponse(BaseModel):
    video_id: str
    text: str
