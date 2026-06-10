"""Transcript segments table reads + row-to-WhisperSegment reconstruction."""

from __future__ import annotations

from collections.abc import Iterable

import aiosqlite

from bibilab.db.connection import get_db


async def get_transcript_segments(source_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT seq, start_s, end_s, speaker, text FROM transcript_segments WHERE source_id = ? ORDER BY seq",
            (source_id,),
        )
        return await cursor.fetchall()


async def get_segments_for_ranges(ranges: list[tuple[str, int, int]]) -> list[aiosqlite.Row]:
    """Fetch transcript segments for many (source_id, seq_start, seq_end) ranges in one query.

    Used by chat top-k reconstruction: each retained chunk contributes one
    range; the union is fetched once and sliced per chunk by the caller.
    """
    if not ranges:
        return []
    clauses = " OR ".join("(source_id = ? AND seq BETWEEN ? AND ?)" for _ in ranges)
    params: list = [val for r in ranges for val in r]
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT source_id, seq, start_s, end_s, speaker, text "
            f"FROM transcript_segments WHERE {clauses} ORDER BY source_id, seq",
            params,
        )
        return await cursor.fetchall()


def rows_to_segments(rows: Iterable[aiosqlite.Row]) -> list:
    """Reconstruct WhisperSegment objects from transcript_segments DB rows."""
    from bibilab.pipeline.transcribe import WhisperSegment

    return [
        WhisperSegment(
            start=r["start_s"],
            end=r["end_s"],
            text=r["text"],
            speaker=r["speaker"],
        )
        for r in rows
    ]
