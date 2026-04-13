import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from bibilab.adapters.base import AuthRequiredError, DownloadError, VideoMeta
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.config import BibilabConfig, get_config
from bibilab.db import create_job, get_list, get_video_statuses
from bibilab.models._enums import VideoStatus
from bibilab.models.ingest import (
    IngestPreviewRequest,
    IngestPreviewResponse,
    IngestUrlRequest,
    IngestUrlResponse,
    PreviewVideo,
    VideoMetadata,
    VideoMetadataMapResponse,
    VideoMetadataRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest/preview")
async def ingest_preview(
    req: IngestPreviewRequest,
    cfg: BibilabConfig = Depends(get_config),
) -> IngestPreviewResponse:
    if await get_list(req.list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")

    try:
        result = BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie).resolve_flat(req.url)
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

    videos = result.videos

    if not videos:
        return IngestPreviewResponse(videos=[])

    statuses = await get_video_statuses([v.video_id for v in videos], req.list_id)

    preview_videos = [
        PreviewVideo(
            video_id=v.video_id,
            title=v.title,
            cover_url=v.cover_url,
            duration_seconds=v.duration_seconds,
            uploader=v.uploader,
            platform=v.platform,
            source_url=v.source_url,
            part_label=v.part_label,
            status=VideoStatus(statuses.get(v.video_id, "new")),
        )
        for v in videos
    ]

    return IngestPreviewResponse(videos=preview_videos)


@router.post("/ingest/preview/metadata")
async def ingest_preview_metadata(
    req: VideoMetadataRequest,
    cfg: BibilabConfig = Depends(get_config),
) -> VideoMetadataMapResponse:
    video_ids = req.video_ids
    if not video_ids:
        return VideoMetadataMapResponse(videos={})

    try:
        metadata_map = await BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie).get_videos_metadata(video_ids)
    except DownloadError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": exc.message},
        ) from exc

    videos = {
        video_id: VideoMetadata(
            title=v.title,
            cover_url=v.cover_url,
            duration_seconds=v.duration_seconds,
            uploader=v.uploader,
            source_url=v.source_url,
        )
        for video_id, v in metadata_map.items()
    }

    return VideoMetadataMapResponse(videos=videos)


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
    ui_lang_header = request.headers.get("X-UI-Lang", "en")
    output_lang = cfg.ai.output_language
    resolved_lang = ui_lang_header if output_lang == "ui" else output_lang

    if await get_list(req.list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")

    statuses = await get_video_statuses([v.video_id for v in req.videos], req.list_id)

    queued: list[str] = []
    skipped: list[str] = []

    for video_in in req.videos:
        status = statuses.get(video_in.video_id, "new")
        if status != "new":
            skipped.append(video_in.video_id)
            continue

        video_meta = VideoMeta(
            video_id=video_in.video_id,
            title=video_in.title,
            platform=video_in.platform,
            source_url=video_in.source_url,
            cover_url=video_in.cover_url,
            duration_seconds=video_in.duration_seconds,
            uploader=video_in.uploader,
        )
        queued.append(await _queue_video(video_meta, req.list_id, ui_lang=resolved_lang))

    return IngestUrlResponse(queued=queued, skipped=skipped)
