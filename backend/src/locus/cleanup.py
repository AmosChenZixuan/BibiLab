import json
import logging
from typing import Any

from locus.config import load_config, locus_home
from locus.pipeline.embed import clear_embeddings_for_video

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

    home = locus_home()
    paths = [
        home / "transcripts" / f"{video_id}.txt",
        home / "notes" / f"{video_id}.md",
        home / "notes" / "attachments" / f"{video_id}_cover.jpg",
    ]

    for path in paths:
        path.unlink(missing_ok=True)

    for path in (home / "downloads").glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)

    clear_embeddings_for_video(video_id, load_config())
    logger.info("Cleaned up artifacts for job %s (%s)", job.get("id", ""), video_id)
