import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, bibilab_home, cover_path, get_config, transcript_path
from bibilab.db import get_source, update_source_digest, update_source_facets
from bibilab.models.sources import SourceContentResponse, SourceFacetsUpdate
from bibilab.pipeline.digest import digest

router = APIRouter()


@router.get("/sources/{source_id}")
async def get_source_content(source_id: str) -> SourceContentResponse:
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    _transcript_path = transcript_path(source["id"])
    transcript = _transcript_path.read_text(encoding="utf-8") if _transcript_path.exists() else ""
    return SourceContentResponse.from_source(source, transcript)


@router.get("/sources/{source_id}/cover")
async def get_source_cover(source_id: str) -> FileResponse:
    _cover = cover_path(source_id)
    if not _cover.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(_cover)


@router.post("/sources/{source_id}/rerun", status_code=200)
async def rerun_source(
    source_id: str, request: Request, cfg: BibilabConfig = Depends(get_config)
) -> SourceContentResponse:
    """Re-run digest on an existing source using its stored transcript."""
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if source["transcript_path"] is None:
        raise HTTPException(status_code=404, detail="Source has no transcript")

    _transcript_path = bibilab_home() / source["transcript_path"]
    try:
        transcript_text = _transcript_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript file not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied reading transcript file")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Error reading transcript file: {exc}")

    video_meta = VideoMeta.from_source(source)

    extraction = await asyncio.to_thread(
        digest,
        transcript_text,
        video_meta,
        cfg.ai,
        cfg.ai.output_language,
        request.headers.get("X-UI-Lang"),
        llm_timeout=cfg.transcription.llm_timeout,
    )
    await update_source_digest(
        source_id,
        extraction.summary,
        extraction.keywords,
        series_name=extraction.series_name,
        sequence_number=extraction.sequence_number,
        sequence_kind=extraction.sequence_kind,
        season_number=extraction.season_number,
    )

    return SourceContentResponse.from_source(source, transcript_text)


@router.patch("/sources/{source_id}/facets", status_code=200)
async def patch_source_facets(source_id: str, body: SourceFacetsUpdate) -> SourceContentResponse:
    """Manually correct series/number/season. Replace semantics; sequence_kind untouched."""
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    fields = {name: getattr(body, name) for name in body.model_fields_set}
    if fields:
        await update_source_facets(source_id, **fields)
        source = await get_source(source_id)

    _transcript_path = transcript_path(source["id"])
    transcript = _transcript_path.read_text(encoding="utf-8") if _transcript_path.exists() else ""
    return SourceContentResponse.from_source(source, transcript)
