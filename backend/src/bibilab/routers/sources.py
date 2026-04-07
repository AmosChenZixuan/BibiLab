import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, bibilab_home, get_config
from bibilab.db import get_source, update_source_digest
from bibilab.models.sources import SourceContentResponse
from bibilab.pipeline.digest import digest

router = APIRouter()


@router.get("/sources/{source_id}")
async def get_source_content(source_id: str) -> SourceContentResponse:
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    transcript_path = bibilab_home() / source["transcript_path"]
    transcript = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
    return SourceContentResponse.from_source(source, transcript)


@router.get("/sources/{source_id}/cover")
async def get_source_cover(source_id: str) -> FileResponse:
    cover_path = bibilab_home() / "covers" / f"{source_id}.jpg"
    if not cover_path.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(cover_path)


@router.post("/sources/{source_id}/rerun", status_code=200)
async def rerun_source(source_id: str, cfg: BibilabConfig = Depends(get_config)) -> SourceContentResponse:
    """Re-run digest on an existing source using its stored transcript."""
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if source["transcript_path"] is None:
        raise HTTPException(status_code=404, detail="Source has no transcript")

    transcript_path = bibilab_home() / source["transcript_path"]
    try:
        transcript_text = transcript_path.read_text(encoding="utf-8")
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
