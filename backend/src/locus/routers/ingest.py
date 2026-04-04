import logging

from fastapi import APIRouter, HTTPException

from locus.adapters.base import AuthRequiredError, DownloadError, VideoMeta
from locus.adapters.bilibili import BilibiliAdapter
from locus.config import load_config
from locus.db import create_job, get_list, get_processed_video_ids
from locus.models.ingest import IngestUrlRequest, IngestUrlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def _queue_video(video: VideoMeta, list_id: str, rerun: bool = False) -> str:
    return await create_job(
        type="ingest",
        meta={
            "video_id": video.video_id,
            "list_id": list_id,
            "title": video.title,
            "cover_url": video.cover_url,
            "duration_seconds": video.duration_seconds,
            "uploader": video.uploader,
            "rerun": rerun,
            "source_url": video.source_url,
            "platform": video.platform,
        },
    )


@router.post("/ingest/url")
async def ingest_url(req: IngestUrlRequest, rerun: bool = False) -> IngestUrlResponse:
    cfg = load_config()

    if await get_list(req.list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")

    adapter = BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)

    try:
        result = adapter.resolve(req.url)
    except AuthRequiredError as exc:
        raise HTTPException(
            status_code=401,
            detail={"message": "Authentication required", "resource_type": exc.resource_type},
        ) from exc
    except DownloadError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": exc.message},
        ) from exc

    videos = [result] if isinstance(result, VideoMeta) else result.videos

    already_done = await get_processed_video_ids([v.video_id for v in videos])
    queued: list[str] = []
    skipped: list[str] = []

    for video in videos:
        if not rerun and video.video_id in already_done:
            logger.info("Skipping already-processed video %s", video.video_id)
            skipped.append(video.video_id)
        else:
            queued.append(await _queue_video(video, req.list_id, rerun=rerun))

    return IngestUrlResponse(queued=queued, skipped=skipped)
