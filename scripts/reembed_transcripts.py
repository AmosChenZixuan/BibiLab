#!/usr/bin/env python3
"""One-shot re-embed all videos using the new multilingual model.

Drops the ChromaDB collection and rebuilds from on-disk transcripts.
No re-download, no re-transcribe.

Usage: python scripts/reembed_transcripts.py
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from bibilab.config import load_config, bibilab_home
from bibilab.db import get_db_path
from bibilab.pipeline.chunk import WhisperSegment, chunk_segments
from bibilab.pipeline.embed import embed_chunks
from bibilab.adapters.base import VideoMeta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_transcript_line(line: str) -> WhisperSegment | None:
    """Parse a single line from write_transcript output: [HH:MM:SS] text"""
    if not line.startswith("["):
        return None
    try:
        time_part, text = line[1:].split("]", 1)
        h, m, s = time_part.split(":")
        start = int(h) * 3600 + int(m) * 60 + int(s)
        text = text.strip()
        if not text:
            return None
        # Placeholder end; corrected in second pass
        return WhisperSegment(start=start, end=start + 30, text=text)
    except Exception:
        return None


def _load_transcript_segments(transcript_path: Path) -> list[WhisperSegment]:
    """Load WhisperSegments from a transcript file written by write_transcript."""
    if not transcript_path.exists():
        return []
    segments = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        seg = _parse_transcript_line(line)
        if seg is not None:
            segments.append(seg)
    # Second pass: set each segment's end to the next segment's start
    for i in range(len(segments) - 1):
        segments[i] = WhisperSegment(
            start=segments[i].start,
            end=segments[i + 1].start,
            text=segments[i].text,
        )
    return segments


def run() -> None:
    cfg = load_config()
    db_path = get_db_path()
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT video_id, list_id, title, platform, source_url, cover_url, duration_seconds, uploader, transcript_path FROM sources WHERE transcript_path IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No sources with transcript_path — nothing to re-embed")
        return

    # Drop Chroma collection first
    import chromadb

    client = chromadb.PersistentClient(path=str(bibilab_home() / "chroma"))
    coll_name = cfg.transcript_collection_name
    try:
        client.delete_collection(coll_name)
        logger.info("Dropped Chroma collection: %s", coll_name)
    except Exception as exc:
        logger.warning("Could not delete collection (may not exist): %s", exc)

    errors = 0
    for (
        video_id,
        list_id,
        title,
        platform,
        source_url,
        cover_url,
        duration_seconds,
        uploader,
        transcript_rel,
    ) in rows:
        transcript_path = bibilab_home() / transcript_rel
        if not transcript_path.exists():
            logger.warning("Transcript missing for %s: %s", video_id, transcript_path)
            errors += 1
            continue

        segments = _load_transcript_segments(transcript_path)
        if not segments:
            logger.warning("No segments parsed for %s", video_id)
            errors += 1
            continue

        # Language hint: Chinese if any non-ASCII in first 200 chars
        sample_text = " ".join(s.text for s in segments[:10])
        language = "zh" if any(ord(c) > 127 for c in sample_text[:200]) else "en"

        chunks = chunk_segments(segments, language=language)
        source = {
            "video_id": video_id,
            "title": title or "unknown",
            "platform": platform,
            "source_url": source_url or "",
            "cover_url": cover_url or "",
            "duration_seconds": duration_seconds or 0,
            "uploader": uploader or "",
        }
        meta = VideoMeta.from_source(source)
        try:
            embed_chunks(chunks, meta, list_id, cfg)
            logger.info("Re-embedded %d chunks for %s", len(chunks), video_id)
        except Exception as exc:
            logger.error("Failed to re-embed %s: %s", video_id, exc)
            errors += 1

    logger.info("Re-embed complete. videos=%d errors=%d", len(rows), errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run()
