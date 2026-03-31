"""Obsidian note rendering and writing."""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from locus.adapters.base import VideoMeta
from locus.config import ObsidianConfig
from locus.pipeline.extract import ExtractionResult

logger = logging.getLogger(__name__)

_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


def _safe_filename(title: str) -> str:
    return _UNSAFE_CHARS.sub("_", title).strip()


def _vault_list_dir(list_name: str, cfg: ObsidianConfig) -> Path:
    return Path(cfg.vault_path) / cfg.locus_folder / list_name


def _download_cover(cover_url: str, dest: Path) -> bool:
    """Download cover image to dest. Returns True on success."""
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
    list_name: str,
    list_id: str,
    cfg: ObsidianConfig,
) -> Path:
    list_dir = _vault_list_dir(list_name, cfg)
    list_dir.mkdir(parents=True, exist_ok=True)

    safe_title = _safe_filename(extraction.title or meta.title)
    note_path = list_dir / f"{safe_title}.md"

    # Cover image
    cover_rel = f"{cfg.locus_folder}/{list_name}/attachments/{meta.video_id}_cover.jpg"
    cover_dest = list_dir / "attachments" / f"{meta.video_id}_cover.jpg"
    has_cover = _download_cover(meta.cover_url, cover_dest)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    frontmatter = (
        "---\n"
        f"locus_id: {meta.video_id}\n"
        f"platform: {meta.platform}\n"
        f"source_url: {meta.source_url}\n"
        f"cover: {cover_rel}\n"
        f"duration: {meta.duration_seconds}\n"
        f"list_id: {list_id}\n"
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


def write_overview_note(
    list_id: str,
    list_name: str,
    videos: list[VideoMeta],
    extraction_results: list[ExtractionResult],
    outline: str,
    cfg: ObsidianConfig,
) -> Path:
    list_dir = _vault_list_dir(list_name, cfg)
    list_dir.mkdir(parents=True, exist_ok=True)
    note_path = list_dir / "_overview.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    frontmatter = (
        "---\n"
        f"locus_list_id: {list_id}\n"
        f"video_count: {len(videos)}\n"
        f"last_updated: {now}\n"
        "---\n"
    )

    video_links = "\n".join(
        f"- [[{_safe_filename(ext.title or m.title)}]]"
        for m, ext in zip(videos, extraction_results)
    )

    body = (
        f"# {list_name} — Overview\n\n" f"## Outline\n{outline}\n\n" f"## Videos\n{video_links}\n"
    )

    tmp = note_path.with_suffix(".tmp")
    tmp.write_text(frontmatter + "\n" + body, encoding="utf-8")
    os.replace(tmp, note_path)
    logger.info("Wrote overview note %s", note_path)
    return note_path
