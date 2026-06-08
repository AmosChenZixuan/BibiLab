"""One-shot backfill: populate `sections` rows for pre-#452 sources.

Run once against the live DB after this issue lands:

    uv run python -m bibilab.scripts.backfill_sections

Post-condition: SELECT COUNT(*) FROM sections == SELECT COUNT(*) FROM sources.
On mismatch the script exits nonzero with a clear error.

DELETE THIS SCRIPT after the verified run (per the project's one-shot rule).
"""

import logging
import sys

from bibilab.db import get_db, get_sections, get_transcript_segments
from bibilab.pipeline.section import derive_sections
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)


async def run_backfill(*, fail_on_mismatch: bool = False) -> None:
    """Populate `sections` rows for sources that have none.

    For each source: derive sections from its segments, assert exactly 1
    section (the corpus invariant for pre-#452 sources), insert that section
    row with summary/keywords copied from the source row (the 1-section
    mirror invariant). No LLM call.
    """
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources")
        sources = list(await cursor.fetchall())

    for source in sources:
        existing = await get_sections(source["id"])
        if existing:
            logger.info(
                "skip source_id=%s: already has %d section row(s)",
                source["id"],
                len(existing),
            )
            continue

        segment_rows = await get_transcript_segments(source["id"])
        if not segment_rows:
            logger.warning("skip source_id=%s: no segments", source["id"])
            continue
        segments = [
            WhisperSegment(
                start=r["start_s"],
                end=r["end_s"],
                text=r["text"],
                speaker=r["speaker"],
            )
            for r in segment_rows
        ]
        sections = derive_sections(segments)
        if len(sections) != 1:
            logger.warning(
                "skip source_id=%s: derive_sections returned %d (not 1); re-ingest to section",
                source["id"],
                len(sections),
            )
            continue

        sec = sections[0]
        async with get_db() as db:
            await db.execute(
                "INSERT INTO sections (source_id, seq, seg_start, seg_end, "
                "token_count, timestamp_start, timestamp_end, summary, keywords) "
                "VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source["id"],
                    sec.seg_start,
                    sec.seg_end,
                    sec.token_count,
                    sec.timestamp_start,
                    sec.timestamp_end,
                    source["summary"],
                    source["keywords"],  # already JSON-encoded in the DB
                ),
            )
            await db.commit()
        logger.info("backfilled source_id=%s", source["id"])

    async with get_db() as db:
        cur_sources = await (await db.execute("SELECT COUNT(*) FROM sources")).fetchone()
        cur_sections = await (await db.execute("SELECT COUNT(*) FROM sections")).fetchone()
    n_sources = cur_sources[0]
    n_sections = cur_sections[0]
    if n_sources != n_sections:
        msg = (
            f"backfill post-condition failed: sources count {n_sources} != "
            f"sections count {n_sections}. Investigate skipped sources above."
        )
        if fail_on_mismatch:
            sys.exit(msg)
        logger.warning(msg)
        return
    logger.info("backfill complete: %d sources, %d sections", n_sources, n_sections)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_backfill(fail_on_mismatch=True))
