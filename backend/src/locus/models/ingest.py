from pydantic import BaseModel


class IngestUrlRequest(BaseModel):
    list_id: str
    url: str


class IngestUrlResponse(BaseModel):
    queued: list[str]  # job IDs created
    skipped: list[str]  # video_ids already in processing_log
