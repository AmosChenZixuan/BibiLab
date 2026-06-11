from __future__ import annotations

from pydantic import BaseModel, Field

from bibilab.db import get_sections, get_segments_for_ranges


class Claim(BaseModel):
    source_id: str
    section_seq: int
    text: str
    snippet: str = ""
    entities: list[str] = Field(default_factory=list)
    is_cause: bool = False   # explains WHY some event happened
    has_time: bool = False   # carries an explicit order / time marker


async def load_spans(source_id: str) -> list[dict]:
    """One span per section: {source_id, section_seq, text}. Reads each section's
    segments once via the shared range helper."""
    sections = await get_sections(source_id)
    spans: list[dict] = []
    for sec in sections:
        rows = await get_segments_for_ranges([(source_id, sec["seg_start"], sec["seg_end"])])
        text = " ".join(r["text"] for r in rows)
        spans.append({"source_id": source_id, "section_seq": sec["seq"], "text": text})
    return spans
