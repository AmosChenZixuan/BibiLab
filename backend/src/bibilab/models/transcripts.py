from pydantic import BaseModel


class TranscriptResponse(BaseModel):
    video_id: str
    total_lines: int
    offset: int
    lines: list[str]
