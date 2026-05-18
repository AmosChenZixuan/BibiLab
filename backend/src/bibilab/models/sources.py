import json
from datetime import datetime

from pydantic import BaseModel


class SourceContentResponse(BaseModel):
    id: str
    video_id: str
    platform: str
    title: str
    source_url: str
    duration_seconds: int
    uploader: str
    language: str | None
    processed_at: datetime | None
    summary: str
    keywords: list[str]
    cover_url: str | None
    transcript: str
    series_name: str | None = None
    sequence_number: int | None = None
    sequence_kind: str | None = None
    season_number: int | None = None

    @classmethod
    def from_source(cls, source: dict, transcript: str) -> "SourceContentResponse":
        keywords = source["keywords"]
        if not isinstance(keywords, list):
            keywords = json.loads(keywords or "[]")
        processed_at_str = source["processed_at"]
        processed_at: datetime | None = None
        if processed_at_str:
            try:
                processed_at = datetime.fromisoformat(processed_at_str)
            except (ValueError, TypeError):
                processed_at = None
        return cls(
            id=source["id"],
            video_id=source["video_id"],
            platform=source["platform"],
            title=source["title"],
            source_url=source["source_url"],
            duration_seconds=source["duration_seconds"],
            uploader=source["uploader"],
            language=source["language"],
            processed_at=processed_at,
            summary=source["summary"],
            keywords=keywords,
            cover_url=source["cover_url"],
            transcript=transcript,
            series_name=source["series_name"],
            sequence_number=source["sequence_number"],
            sequence_kind=source["sequence_kind"],
            season_number=source["season_number"],
        )
