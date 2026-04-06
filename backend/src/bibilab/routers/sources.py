import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bibilab.config import bibilab_home
from bibilab.db import get_source
from bibilab.models.sources import SourceContentResponse

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
