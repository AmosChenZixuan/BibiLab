import logging

from fastapi import APIRouter, HTTPException

from locus.adapters.base import AuthRequiredError, VideoMeta
from locus.adapters.bilibili import BilibiliAdapter
from locus.config import load_config
from locus.db import create_job, get_db
from locus.models.ingest import IngestUrlRequest, IngestUrlResponse
from locus.vault import get_list_by_id

router = APIRouter()
logger = logging.getLogger(__name__)

_PLATFORM_URL_TEMPLATES: dict[str, str] = {
    "bilibili": "https://www.bilibili.com/video/{video_id}",
}


def _canonical_url(video_id: str, platform: str) -> str:
    template = _PLATFORM_URL_TEMPLATES.get(platform)
    if template is None:
        raise ValueError(f"No URL template for platform {platform!r}")
    return template.format(video_id=video_id)


def _list_exists(list_id: str, vault_cfg) -> bool:
    if not vault_cfg.vault_path:
        return False
    return get_list_by_id(list_id, vault_cfg) is not None


async def _already_processed_batch(video_ids: list[str]) -> set[str]:
    if not video_ids:
        return set()
    placeholders = ",".join("?" * len(video_ids))
    async with get_db() as db:
        async with db.execute(
            f"SELECT video_id FROM processing_log WHERE video_id IN ({placeholders})",
            video_ids,
        ) as cur:
            rows = await cur.fetchall()
    return {row["video_id"] for row in rows}


async def _queue_video(video: VideoMeta, list_id: str, rerun: bool = False) -> str:
    return await create_job(
        type="video",
        source_url=video.source_url,
        platform=video.platform,
        meta={
            "video_id": video.video_id,
            "list_id": list_id,
            "title": video.title,
            "cover_url": video.cover_url,
            "duration_seconds": video.duration_seconds,
            "uploader": video.uploader,
            "rerun": rerun,
        },
    )


@router.post("/ingest/url")
async def ingest_url(req: IngestUrlRequest) -> IngestUrlResponse:
    cfg = load_config()

    # Validate vault path before queuing — fail fast rather than mid-pipeline
    if not cfg.obsidian.vault_path:
        raise HTTPException(
            status_code=400,
            detail="obsidian.vault_path is not configured. Set it via PUT /config.",
        )

    if not _list_exists(req.list_id, cfg.obsidian):
        raise HTTPException(status_code=404, detail="List not found")

    adapter = BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)

    try:
        result = adapter.resolve(req.url)
    except AuthRequiredError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Authentication required",
                "resource_type": exc.resource_type,
            },
        ) from exc

    videos = [result] if isinstance(result, VideoMeta) else result.videos

    already_done = await _already_processed_batch([v.video_id for v in videos])
    queued: list[str] = []
    skipped: list[str] = []

    for video in videos:
        if video.video_id in already_done:
            logger.info("Skipping already-processed video %s", video.video_id)
            skipped.append(video.video_id)
        else:
            queued.append(await _queue_video(video, req.list_id))

    return IngestUrlResponse(queued=queued, skipped=skipped)


@router.post("/ingest/rerun/{video_id}")
async def ingest_rerun(video_id: str) -> IngestUrlResponse:
    async with get_db() as db:
        async with db.execute("SELECT * FROM processing_log WHERE video_id=?", (video_id,)) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Video not in processing log")

    row = dict(row)
    platform = row["platform"]
    try:
        source_url = _canonical_url(video_id, platform)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video = VideoMeta(
        video_id=video_id,
        title="",  # will be re-fetched by pipeline
        platform=platform,
        source_url=source_url,
        cover_url="",
        duration_seconds=0,
        uploader="",
    )
    job_id = await _queue_video(video, list_id=row.get("list_id", ""), rerun=True)
    return IngestUrlResponse(queued=[job_id] if job_id else [], skipped=[])
