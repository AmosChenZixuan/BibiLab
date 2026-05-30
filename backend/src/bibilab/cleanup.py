import json
import logging
from typing import Any

from bibilab.config import bibilab_home, load_config
from bibilab.pipeline.embed import clear_embeddings_for_video, clear_fts_for_video_sync

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

    # Clean up cover image using source_id from meta
    source_id = meta.get("source_id")
    if isinstance(source_id, str) and source_id:
        cover_path = home / "covers" / f"{source_id}.jpg"
        cover_path.unlink(missing_ok=True)

    clear_embeddings_for_video(video_id, load_config())
    clear_fts_for_video_sync(video_id)
    logger.info("Cleaned up artifacts for job %s (%s)", job.get("id", ""), video_id)
