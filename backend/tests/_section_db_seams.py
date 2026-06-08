"""Test-only seams for the `sections` table.

These helpers used to live in `bibilab.db` as `write_sections` and
`get_sections` — non-production seams whose only callers were the unit
tests and the (now-removed) one-shot backfill script. They live here so
production modules don't carry test-only code.

The **production** write path is `bibilab.db.write_source_with_segments`,
which writes sections atomically with the source row and segments in one
transaction. Do not import these helpers from a production code path —
they open their own connection and commit independently.
"""

import aiosqlite

from bibilab.db import get_db
from bibilab.pipeline.section import Section


async def write_sections(source_id: str, sections: list[Section]) -> None:
    """Test helper: replace all section rows for a source (DELETE + INSERT).

    Opens its own connection via `get_db()` and commits. Non-production seam;
    use `bibilab.db.write_source_with_segments(sections=...)` for atomic
    source + segments + sections writes.
    """
    async with get_db() as db:
        await db.execute("DELETE FROM sections WHERE source_id = ?", (source_id,))
        await db.executemany(
            "INSERT INTO sections (source_id, seq, seg_start, seg_end, "
            "token_count, timestamp_start, timestamp_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (source_id, i, s.seg_start, s.seg_end, s.token_count, s.timestamp_start, s.timestamp_end)
                for i, s in enumerate(sections)
            ],
        )
        await db.commit()


async def get_sections(source_id: str) -> list[aiosqlite.Row]:
    """Test helper: fetch a source's section rows in seq order."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, source_id, seq, seg_start, seg_end, "
            "token_count, timestamp_start, timestamp_end, summary, keywords "
            "FROM sections WHERE source_id = ? ORDER BY seq",
            (source_id,),
        )
        return await cursor.fetchall()
