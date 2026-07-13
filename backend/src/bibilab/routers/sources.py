from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from bibilab.config import cover_path
from bibilab.db import (
    create_job,
    get_pending_jobs,
    get_sections,
    get_source,
    parse_job_meta,
    source_has_segments,
    update_source_facets,
)
from bibilab.models.jobs import JobType
from bibilab.models.sources import SectionListItem, SourceContentResponse, SourceFacetsUpdate
from bibilab.pipeline._shared import UI_LANG_HEADER
from bibilab.pipeline.transcribe import load_transcript_text

router = APIRouter()


@router.get("/sources/{source_id}")
async def get_source_content(source_id: str, include_time: bool = True) -> SourceContentResponse:
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    transcript = await load_transcript_text(source["id"], include_time=include_time)
    return SourceContentResponse.from_source(source, transcript)


@router.get("/sources/{source_id}/cover")
async def get_source_cover(source_id: str) -> FileResponse:
    _cover = cover_path(source_id)
    if not _cover.exists():
        raise HTTPException(status_code=404, detail="Cover not found")
    return FileResponse(_cover)


@router.get("/sources/{source_id}/sections")
async def get_source_sections(source_id: str) -> list[SectionListItem]:
    """List a source's section rows in seq order."""
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    rows = await get_sections(source_id)
    return [SectionListItem.from_row(r) for r in rows]


@router.post("/sources/{source_id}/rerun", status_code=202)
async def rerun_source(source_id: str, request: Request) -> dict:
    """Re-run digest on an existing source using its stored transcript.

    Creates a background digest job and returns immediately with the job_id.
    """
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if not await source_has_segments(source_id):
        raise HTTPException(status_code=404, detail="Source has no transcript")

    # Dedup: reject if a non-terminal digest job already exists for this source
    pending = await get_pending_jobs()
    for job in pending:
        if job["type"] == JobType.DIGEST:
            meta = parse_job_meta(job)
            if meta.get("source_id") == source_id:
                raise HTTPException(status_code=409, detail="Digest already in progress")

    ui_lang = request.headers.get(UI_LANG_HEADER)
    job_id = await create_job(
        type=JobType.DIGEST,
        meta={
            "source_id": source_id,
            "list_id": source["list_id"],
            "source_title": source["title"],
            "ui_lang": ui_lang,
        },
    )
    return {"job_id": job_id}


@router.patch("/sources/{source_id}/facets", status_code=204)
async def patch_source_facets(source_id: str, body: SourceFacetsUpdate) -> Response:
    """Manually correct series/number/season. Replace semantics.

    Returns 204 — the client refetches via GET /sources/{id}; echoing the
    transcript-laden SourceContentResponse here would be read twice and discarded.
    """
    source = await get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    fields = {name: getattr(body, name) for name in body.model_fields_set}
    if fields:
        try:
            await update_source_facets(source_id, **fields)
        except LookupError:
            raise HTTPException(status_code=404, detail="Source not found")

    return Response(status_code=204)
