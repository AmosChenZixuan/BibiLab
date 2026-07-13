import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from bibilab.adapters import get_adapter_for_platform, get_adapter_for_url
from bibilab.adapters.base import (
    AuthRequiredError,
    DownloadError,
    UnsupportedPlatformError,
    VideoMeta,
)
from bibilab.config import BibilabConfig, get_config
from bibilab.db import create_job, get_list
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
from bibilab.models.jobs import JobType
from bibilab.pipeline._shared import UI_LANG_HEADER, resolve_response_language
from bibilab.routers._model_gate import require_models_present
from bibilab.video_status import get_video_statuses

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
        adapter = get_adapter_for_url(req.url, cfg)
        # yt-dlp resolution is a multi-second blocking HTTP call — off the loop,
        # or every concurrent handler (chat SSE, polls) stalls behind it.
        result = await asyncio.to_thread(adapter.resolve_flat, req.url)
    except UnsupportedPlatformError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
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
        adapter = get_adapter_for_platform(req.platform, cfg)
    except UnsupportedPlatformError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    try:
        metadata_map, expanded = await adapter.get_videos_metadata(video_ids)
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
            part_label=v.part_label,
        )
        for video_id, v in metadata_map.items()
    }

    return VideoMetadataMapResponse(videos=videos, expanded=expanded)


async def _queue_video(
    video: VideoMeta,
    list_id: str,
    ui_lang: str = "en",
) -> str:
    return await create_job(
        type=JobType.INGEST,
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
    resolved_lang = resolve_response_language(cfg.ai, request.headers.get(UI_LANG_HEADER, "en"))

    if await get_list(req.list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")

    require_models_present(cfg)

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
