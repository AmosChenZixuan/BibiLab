import json
from datetime import datetime

import aiosqlite
from pydantic import BaseModel, field_validator


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
    cover_url: str | None
    transcript: str
    series_name: str | None = None
    sequence_number: int | None = None
    season_number: int | None = None

    @classmethod
    def from_source(cls, source: dict, transcript: str) -> "SourceContentResponse":
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
            cover_url=source["cover_url"],
            transcript=transcript,
            series_name=source["series_name"],
            sequence_number=source["sequence_number"],
            season_number=source["season_number"],
        )


class SectionListItem(BaseModel):
    """Projected section row for GET /sources/{id}/sections.

    `section_id` is the row's primary key projected as a string — the
    chat citation-jump path needs it to resolve a cited section without
    relying on the chunk-anchored `timestamp_start` (which can land at
    a section boundary or outside any section's range). The other
    internal columns (source_id, seg_start, seg_end, token_count) stay
    hidden — they're an implementation detail of the rerun path.
    """

    section_id: str
    seq: int
    summary: str
    keywords: list[str]
    timestamp_start: float
    timestamp_end: float

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "SectionListItem":
        """Project a sections-table row to the API response shape.

        Every section row carries a summary/keywords (written with the
        section in one transaction). `id` is projected as `section_id`
        (stringified) for the chat citation-jump match key; the other
        internal columns (source_id, seg_start, seg_end, token_count)
        are not exposed.
        """
        return cls(
            section_id=str(row["id"]),
            seq=row["seq"],
            summary=row["summary"],
            keywords=json.loads(row["keywords"]),
            timestamp_start=row["timestamp_start"],
            timestamp_end=row["timestamp_end"],
        )


class SourceFacetsUpdate(BaseModel):
    """Manual facet edit.

    `model_fields_set` lets the router distinguish an absent key (leave the
    column alone) from an explicit null (clear it). Invalid ints raise
    ValueError -> FastAPI 422 (a typed value is deliberate, never degraded).
    """

    series_name: str | None = None
    sequence_number: int | None = None
    season_number: int | None = None

    @field_validator("sequence_number", "season_number", mode="before")
    @classmethod
    def _ints(cls, v: object) -> int | None:
        from bibilab.pipeline.digest import parse_facet_int

        return parse_facet_int(v)

    @field_validator("series_name", mode="before")
    @classmethod
    def _series(cls, v: object) -> str | None:
        from bibilab.pipeline.digest import clean_str_facet

        return clean_str_facet(v)
