import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bibilab.adapters.base import VideoMeta
from bibilab.config import bibilab_home, load_config
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
    return SourceContentResponse(
        id=source["id"],
        video_id=source["video_id"],
        platform=source["platform"],
        title=source["title"],
        source_url=source["source_url"],
        duration_seconds=source["duration_seconds"],
        uploader=source["uploader"],
        language=source["language"],
        processed_at=source["processed_at"] or "",
        summary=source["summary"],
        keywords=json.loads(source["keywords"] or "[]"),
        cover_url=source["cover_url"],
        transcript=transcript,
        settings_snapshot=json.loads(source["settings_snapshot"] or "{}"),
    )


@router.get("/sources/{source_id}/cover")
async def get_source_cover(source_id: str) -> FileResponse:
    cover_path = bibilab_home() / "covers" / f"{source_id}.jpg"
    if not cover_path.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(cover_path)


@router.post("/sources/{source_id}/rerun", status_code=200)
async def rerun_source(source_id: str) -> SourceContentResponse:
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

    video_meta = VideoMeta(
        video_id=source["video_id"],
        title=source["title"],
        platform=source["platform"],
        source_url=source["source_url"],
        cover_url=source["cover_url"] or "",
        duration_seconds=source["duration_seconds"],
        uploader=source["uploader"],
    )

    cfg = load_config()
    extraction = await asyncio.to_thread(
        digest,
        transcript_text,
        video_meta,
        cfg.ai,
        cfg.ai.output_language,
    )
    await update_source_digest(source_id, extraction.summary, extraction.keywords)

    return SourceContentResponse(
        id=source["id"],
        video_id=source["video_id"],
        platform=source["platform"],
        title=source["title"],
        source_url=source["source_url"],
        duration_seconds=source["duration_seconds"],
        uploader=source["uploader"],
        language=source["language"],
        processed_at=source["processed_at"] or "",
        summary=extraction.summary,
        keywords=extraction.keywords,
        cover_url=source["cover_url"],
        transcript=transcript_text,
        settings_snapshot=json.loads(source["settings_snapshot"] or "{}"),
    )
