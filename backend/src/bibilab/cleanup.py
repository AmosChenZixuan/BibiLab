import json
import logging
from typing import Any

from bibilab.config import bibilab_home, load_config
from bibilab.db import source_exists_sync
from bibilab.pipeline.embed import clear_embeddings_for_source, clear_fts_for_source_sync

logger = logging.getLogger(__name__)


def _parse_meta(job: dict[str, Any]) -> dict[str, Any]:
    meta = job.get("meta", {})
    if isinstance(meta, str):
        return json.loads(meta or "{}")
    return meta


def cleanup_job_artifacts(job: dict[str, Any]) -> None:
    if job.get("type") != "ingest" or job.get("status") == "done":
        return

    meta = _parse_meta(job)
    video_id = meta.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return

    home = bibilab_home()

    for path in (home / "downloads").glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)

    # Clean up cover image and embeddings using source_id from meta.
    # A committed source row means the ingest reached persist (Stage 5) and its
    # cover/embeddings/FTS are live — never purge them as a partial-failure cleanup,
    # even if the job later failed before reaching DONE.
    source_id = meta.get("source_id")
    if isinstance(source_id, str) and source_id and not source_exists_sync(source_id):
        cover_path = home / "covers" / f"{source_id}.jpg"
        cover_path.unlink(missing_ok=True)
        clear_embeddings_for_source(source_id, load_config())
        clear_fts_for_source_sync(source_id)
        logger.info("Cleaned up artifacts for job %s (source %s)", job.get("id", ""), source_id)
