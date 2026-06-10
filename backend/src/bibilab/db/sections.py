"""Sections table CRUD + row-to-Section reconstruction."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

import aiosqlite

from bibilab.db.connection import get_db

logger = logging.getLogger(__name__)


async def update_section_summaries(
    source_id: str,
    section_digests: list[tuple[int, str, list[str]]],
) -> None:
    """UPDATE existing section rows by seq. Rerun path.

    Each tuple is (seq, summary, keywords). The caller is the worker's
    rerun path, which built the tuples by zipping `get_sections(source_id)`
    rows with the new `SectionDigest`s, so seqs are unique and exist by
    construction. Missing or duplicate seqs are caller bugs; this helper
    does not pre-validate.
    """
    if not section_digests:
        logger.warning("update_section_summaries: empty input for source_id=%s (no-op)", source_id)
        return
    async with get_db() as db:
        await db.executemany(
            "UPDATE sections SET summary=?, keywords=? WHERE source_id=? AND seq=?",
            [(summary, json.dumps(keywords), source_id, seq) for seq, summary, keywords in section_digests],
        )
        await db.commit()


async def get_sections(source_id: str) -> list[aiosqlite.Row]:
    """Fetch all section rows for a source, ordered by seq.

    Returns all 10 columns: id, source_id, seq, seg_start, seg_end,
    token_count, timestamp_start, timestamp_end, summary, keywords.
    Callers (the API, the rerun path) project what they need.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, source_id, seq, seg_start, seg_end, "
            "token_count, timestamp_start, timestamp_end, summary, keywords "
            "FROM sections WHERE source_id = ? ORDER BY seq",
            (source_id,),
        )
        return await cursor.fetchall()


async def get_section_ranges(source_id: str) -> list[aiosqlite.Row]:
    """Return a source's sections ordered by seq.

    Each row carries ``seq, seg_start, seg_end, token_count, timestamp_start,
    timestamp_end`` — the verbatim-reconstruction contract used by the
    artifact batched-refine and the chat surface. Pure read;
    returns ``[]`` when the source has no section rows (the caller decides
    whether that's a fail-loud error, per its policy).
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT seq, seg_start, seg_end, token_count, "
            "timestamp_start, timestamp_end "
            "FROM sections WHERE source_id = ? ORDER BY seq",
            (source_id,),
        )
        return await cursor.fetchall()


def rows_to_sections(rows: Iterable[aiosqlite.Row]) -> list:
    """Reconstruct Section objects from sections DB rows."""
    from bibilab.pipeline.section import Section

    return [
        Section(
            seg_start=r["seg_start"],
            seg_end=r["seg_end"],
            token_count=r["token_count"],
            timestamp_start=r["timestamp_start"] or 0.0,
            timestamp_end=r["timestamp_end"] or 0.0,
        )
        for r in rows
    ]
