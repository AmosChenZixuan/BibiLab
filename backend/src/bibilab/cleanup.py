import logging
from pathlib import Path
from typing import Any

from bibilab.config import bibilab_home, downloads_dir
from bibilab.db import parse_job_meta, source_exists_sync
from bibilab.models.jobs import JobType
from bibilab.pipeline.embed import clear_embeddings_for_source, clear_fts_for_source_sync

logger = logging.getLogger(__name__)

# Hardcoded cap for the download-stage cache (LRU by mtime). 10 GB is enough
# for a typical ingest window without runaway disk growth on long-running
# installs. Override in tests via monkeypatch on this module attribute.
CACHE_MAX_BYTES = 10 * 1024**3


def purge_download_files(video_id: str) -> None:
    """Remove any downloads/{video_id}.* files, including yt-dlp .part residue.

    Used as partial-failure cleanup and as pre-download hygiene, so a new
    download never resumes onto bytes left by a previous failed/corrupt attempt.
    """
    for path in downloads_dir().glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)


def _find_cached_video(video_id: str) -> Path | None:
    """Locate a usable cached download for `video_id`, or None on miss.

    A cache entry is any `downloads/{video_id}.{ext}` file with size > 0,
    excluding yt-dlp `.part` residue from in-flight downloads.
    """
    for path in downloads_dir().glob(f"{video_id}.*"):
        if path.name.endswith(".part"):
            continue
        if path.stat().st_size <= 0:
            continue
        return path
    return None


def cleanup_job_artifacts(job: dict[str, Any]) -> None:
    if job.get("type") != JobType.INGEST or job.get("status") == "done":
        return

    meta = parse_job_meta(job)
    video_id = meta.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return

    home = bibilab_home()

    # Intentionally no purge of downloads/{video_id}.* here: the failure path
    # is the same place use case A (recovery after a processing-stage failure)
    # needs the bytes to survive so re-ingest can hit the cache.

    # Clean up cover image and embeddings using source_id from meta.
    # A committed source row means the ingest reached persist (Stage 5) and its
    # cover/embeddings/FTS are live — never purge them as a partial-failure cleanup,
    # even if the job later failed before reaching DONE.
    source_id = meta.get("source_id")
    if isinstance(source_id, str) and source_id and not source_exists_sync(source_id):
        cover_path = home / "covers" / f"{source_id}.jpg"
        cover_path.unlink(missing_ok=True)
        clear_embeddings_for_source(source_id)
        clear_fts_for_source_sync(source_id)
        logger.info("Cleaned up artifacts for job %s (source %s)", job.get("id", ""), source_id)
