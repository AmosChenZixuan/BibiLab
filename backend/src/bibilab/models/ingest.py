from pydantic import BaseModel


class IngestUrlRequest(BaseModel):
    list_id: str
    url: str
    source_id: str | None = None
    stages: list[str] | None = None


class IngestUrlResponse(BaseModel):
    queued: list[str]
    skipped: list[str]
