"""Note writing - writes to ~/.bibilab/notes/{video_id}.md."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bibilab.adapters.base import VideoMeta
from bibilab.config import bibilab_home
from bibilab.pipeline.extract import ExtractionResult

logger = logging.getLogger(__name__)


def _download_cover(cover_url: str, dest: Path) -> bool:
    if not cover_url:
        return False
    try:
        with httpx.stream("GET", cover_url, timeout=30, follow_redirects=True) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Cover download failed for %s: %s", cover_url, exc)
        return False


def write_video_note(
    meta: VideoMeta,
    extraction: ExtractionResult,
    list_id: str,
) -> Path:
    notes_dir = bibilab_home() / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    note_path = notes_dir / f"{meta.video_id}.md"

    cover_dest = notes_dir / "attachments" / f"{meta.video_id}_cover.jpg"
    has_cover = _download_cover(meta.cover_url, cover_dest)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    frontmatter = (
        "---\n"
        f"video_id: {meta.video_id}\n"
        f"platform: {meta.platform}\n"
        f"source_url: {meta.source_url}\n"
        f"list_id: {list_id}\n"
        f"duration: {meta.duration_seconds}\n"
        f"processed_at: {now}\n"
        "---\n"
    )

    cover_line = f"![cover](attachments/{meta.video_id}_cover.jpg)\n\n" if has_cover else ""
    key_points_md = "\n".join(f"- {kp.timestamp} {kp.text}" for kp in extraction.key_points)
    body = (
        f"# {extraction.title or meta.title}\n\n"
        f"{cover_line}"
        f"## Summary\n{extraction.summary}\n\n"
        f"## Key Points\n{key_points_md}\n"
    )

    tmp = note_path.with_suffix(".tmp")
    tmp.write_text(frontmatter + "\n" + body, encoding="utf-8")
    os.replace(tmp, note_path)
    logger.info("Wrote note %s", note_path)
    return note_path
