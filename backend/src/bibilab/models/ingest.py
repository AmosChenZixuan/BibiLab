from pydantic import BaseModel


class IngestUrlRequest(BaseModel):
    list_id: str
    url: str


class IngestUrlResponse(BaseModel):
    queued: list[str]
    skipped: list[str]
