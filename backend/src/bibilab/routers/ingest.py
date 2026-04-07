import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from bibilab.adapters.base import AuthRequiredError, DownloadError, VideoMeta
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.config import BibilabConfig, get_config
from bibilab.db import create_job, get_list, get_processed_video_ids
from bibilab.models.ingest import IngestUrlRequest, IngestUrlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def _queue_video(
    video: VideoMeta,
    list_id: str,
    ui_lang: str = "en",
) -> str:
    return await create_job(
        type="ingest",
        meta={
            "video_id": video.video_id,
            "list_id": list_id,
            "title": video.title,
            "cover_url": video.cover_url,
            "duration_seconds": video.duration_seconds,
            "uploader": video.uploader,
            "source_url": video.source_url,
            "platform": video.platform,
            "ui_lang": ui_lang,
        },
    )


@router.post("/ingest/url")
async def ingest_url(
    req: IngestUrlRequest,
    request: Request,
    cfg: BibilabConfig = Depends(get_config),
) -> IngestUrlResponse:

    # Resolve UI language
    ui_lang_header = request.headers.get("X-UI-Lang", "en")
    output_lang = cfg.ai.output_language
    if output_lang == "ui":
        resolved_lang = ui_lang_header
    else:
        resolved_lang = output_lang

    if await get_list(req.list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")

    try:
        result = BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie).resolve(req.url)
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

    already_done = await get_processed_video_ids([v.video_id for v in videos], req.list_id)
    queued: list[str] = []
    skipped: list[str] = []

    for video in videos:
        if video.video_id in already_done:
            skipped.append(video.video_id)
            continue
        queued.append(
            await _queue_video(
                video,
                req.list_id,
                ui_lang=resolved_lang,
            )
        )

    return IngestUrlResponse(queued=queued, skipped=skipped)
