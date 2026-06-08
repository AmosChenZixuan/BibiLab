"""Dev one-shot: seed a synthetic multi-section source for end-to-end verification of multi-section rendering.

Concatenates the K=6 longest existing sources' transcript segments with
continuous timestamps + a 5s seam pause, then runs the full ingest
pipeline (sections, chunks, embed, digest). The synthetic source is
indistinguishable from a real multi-section one.

Run once against a dev DB:

    uv run python -m bibilab.scripts.seed_long_source

DELETE THIS SCRIPT after the chat goes section-grained follow-up verifies the synthetic source (per the
project's one-shot rule).
"""

import asyncio
import logging
import sys
import uuid

import aiosqlite

from bibilab.adapters.base import VideoMeta
from bibilab.config import load_config
from bibilab.db import (
    bootstrap_db,
    delete_source,
    get_db_path,
    write_source_with_segments,
)
from bibilab.pipeline.digest import digest_sections
from bibilab.pipeline.embed import embed_chunks
from bibilab.pipeline.section import chunk_by_sections, derive_sections, section_texts
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

K = 6
SEAM_GAP_S = 5.0


async def _load_synthetic_segments() -> tuple[list[WhisperSegment], list[str], list[aiosqlite.Row]]:
    """Concatenate K longest sources' segments with continuous timestamps."""
    db_path = get_db_path()
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        cur = await conn.execute(
            "SELECT s.id, s.title, s.video_id, s.platform, s.list_id, s.uploader, "
            "s.source_url, s.cover_url, s.duration_seconds, s.whisper_model, s.ai_model, "
            "s.settings_snapshot, s.series_name, s.sequence_number, s.season_number, "
            "s.summary, s.keywords, s.language "
            "FROM sources s "
            "JOIN transcript_segments t ON t.source_id=s.id "
            "GROUP BY s.id ORDER BY COUNT(t.id) DESC LIMIT ?",
            (K,),
        )
        ref_rows = list(await cur.fetchall())
        if not ref_rows:
            return [], [], []
        segs: list[WhisperSegment] = []
        titles: list[str] = []
        offset = 0.0
        for r in ref_rows:
            titles.append(r["title"])
            if segs:
                offset += SEAM_GAP_S
            seg_cur = await conn.execute(
                "SELECT start_s, end_s, text, speaker FROM transcript_segments WHERE source_id=? ORDER BY seq",
                (r["id"],),
            )
            seg_rows = list(await seg_cur.fetchall())
            for sr in seg_rows:
                segs.append(
                    WhisperSegment(
                        start=offset + sr["start_s"],
                        end=offset + sr["end_s"],
                        text=sr["text"],
                        speaker=sr["speaker"],
                    )
                )
            if seg_rows:
                offset = segs[-1].end
        return segs, titles, ref_rows
    finally:
        await conn.close()


async def run_seed() -> None:
    cfg = load_config()
    await bootstrap_db()

    if not get_db_path().exists():
        sys.exit(f"DB not found at {get_db_path()}; ingest at least one source first")

    segs, titles, ref_rows = await _load_synthetic_segments()
    if len(segs) < 2:
        sys.exit("not enough segments to seed (need at least 2)")

    synthetic_source_id = str(uuid.uuid4())
    list_id = ref_rows[0]["list_id"]
    first = ref_rows[0]

    synthetic_meta = VideoMeta(
        video_id=f"synthetic-{synthetic_source_id[:8]}",
        title=f"[SEEDED] concatenated {' / '.join(t[:30] for t in titles[:3])}",
        platform=first["platform"],
        source_url=first["source_url"],
        cover_url="",
        duration_seconds=int(segs[-1].end),
        uploader=first["uploader"] or "synthetic",
    )

    sections = derive_sections(segs)
    chunks = chunk_by_sections(segs, sections, language=first["language"] or "zh")
    section_texts_list = section_texts(segs, sections)

    extraction, section_digests = digest_sections(
        section_texts_list,
        synthetic_meta,
        cfg.ai,
        cfg.ai.output_language,
        None,
    )

    await asyncio.to_thread(embed_chunks, chunks, synthetic_source_id, synthetic_meta, list_id)

    try:
        await write_source_with_segments(
            segments=segs,
            sections=sections,
            section_digests=section_digests,
            source_id=synthetic_source_id,
            video_id=synthetic_meta.video_id,
            platform=synthetic_meta.platform,
            list_id=list_id,
            title=synthetic_meta.title,
            summary=extraction.summary,
            keywords=extraction.keywords,
            cover_url=None,
            source_url=synthetic_meta.source_url,
            duration_seconds=synthetic_meta.duration_seconds,
            uploader=synthetic_meta.uploader,
            language=first["language"],
            whisper_model=cfg.transcription.model,
            ai_model=cfg.ai.model,
            settings_snapshot=cfg.model_dump(),
            series_name=extraction.series_name,
            sequence_number=extraction.sequence_number,
            season_number=extraction.season_number,
        )
    except Exception:
        # Roll back the synthetic source on failure (embed_chunks leaves chroma
        # rows behind, but those are harmless and overwritten on next ingest).
        await delete_source(synthetic_source_id)
        raise

    logger.info(
        "seeded synthetic source: id=%s, sections=%d, chunks=%d",
        synthetic_source_id,
        len(sections),
        len(chunks),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_seed())
