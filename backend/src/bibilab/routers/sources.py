import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, cover_path, get_config, transcript_path
from bibilab.db import get_source, update_source_digest
from bibilab.models.sources import SourceContentResponse
from bibilab.pipeline.digest import digest

router = APIRouter()


@router.get("/sources/{source_id}")
async def get_source_content(source_id: str) -> SourceContentResponse:
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    _transcript_path = transcript_path(source["video_id"])
    transcript = _transcript_path.read_text(encoding="utf-8") if _transcript_path.exists() else ""
    return SourceContentResponse.from_source(source, transcript)


@router.get("/sources/{source_id}/cover")
async def get_source_cover(source_id: str) -> FileResponse:
    _cover = cover_path(source_id)
    if not _cover.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(_cover)


@router.post("/sources/{source_id}/rerun", status_code=200)
async def rerun_source(source_id: str, cfg: BibilabConfig = Depends(get_config)) -> SourceContentResponse:
    """Re-run digest on an existing source using its stored transcript."""
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if source["transcript_path"] is None:
        raise HTTPException(status_code=404, detail="Source has no transcript")

    _transcript_path = transcript_path(source["video_id"])
    try:
        transcript_text = _transcript_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript file not found")

    video_meta = VideoMeta.from_source(source)

    extraction = await asyncio.to_thread(
        digest,
        transcript_text,
        video_meta,
        cfg.ai,
        cfg.ai.output_language,
    )
    await update_source_digest(source_id, extraction.summary, extraction.keywords)

    return SourceContentResponse.from_source(source, transcript_text)
