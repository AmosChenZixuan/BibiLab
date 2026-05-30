import asyncio
import json
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request

from bibilab.config import BibilabConfig, bibilab_home, cover_path, get_config
from bibilab.db import (
    clear_fts_for_list,
    delete_artifacts_for_list,
    delete_source,
    delete_sources_for_list,
    get_all_lists,
    get_artifacts_for_list,
    get_db,
    get_list,
    get_list_with_display,
    get_source,
    get_sources_for_list,
    update_list_name,
    update_list_thumbnail,
)
from bibilab.db import (
    create_list as db_create_list,
)
from bibilab.db import (
    delete_list as db_delete_list,
)
from bibilab.models.lists import (
    ListCreateRequest,
    ListResponse,
    ListUpdateRequest,
    OverviewResponse,
    SourceResponse,
)
from bibilab.pipeline.embed import clear_embeddings_for_list, clear_embeddings_for_source, clear_fts_for_source_sync

router = APIRouter()

_ACTIVE_JOB_STATUSES = ("queued", "downloading", "transcribing", "processing")


def _purge_source_resources(source_id: str) -> None:
    cover_path(source_id).unlink(missing_ok=True)


async def _build_list_response(row: aiosqlite.Row, request: Request) -> ListResponse:
    thumbnail_url = None
    if row["thumbnail_source_id"]:
        if cover_path(row["thumbnail_source_id"]).exists():
            thumbnail_url = str(request.url_for("get_source_cover", source_id=row["thumbnail_source_id"]))
        else:
            source = await get_source(row["thumbnail_source_id"])
            if source is not None and source["cover_url"]:
                thumbnail_url = source["cover_url"]

    return ListResponse(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        thumbnail_source_id=row["thumbnail_source_id"],
        thumbnail_url=thumbnail_url,
        source_count=row["source_count"],
        updated_at=row["updated_at"],
    )


@router.post("/lists", status_code=201)
async def create_list(req: ListCreateRequest) -> ListResponse:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="List name cannot be empty")
    list_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db_create_list(list_id, name, now)
    return ListResponse(
        id=list_id,
        name=name,
        created_at=now,
        thumbnail_source_id=None,
        thumbnail_url=None,
        source_count=0,
        updated_at=now,
    )


@router.get("/lists")
async def get_lists(request: Request) -> list[ListResponse]:
    rows = await get_all_lists()
    return [await _build_list_response(r, request) for r in rows]


@router.patch("/lists/{list_id}")
async def update_list(list_id: str, req: ListUpdateRequest, request: Request) -> ListResponse:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    if not req.model_fields_set:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "name" in req.model_fields_set:
        if req.name is None or not req.name.strip():
            raise HTTPException(status_code=422, detail="List name cannot be empty")
        await update_list_name(list_id, req.name.strip())

    if "thumbnail_source_id" in req.model_fields_set:
        if req.thumbnail_source_id is not None:
            source = await get_source(req.thumbnail_source_id)
            if source is None or source["list_id"] != list_id:
                raise HTTPException(status_code=422, detail="Thumbnail source must belong to the list")
        await update_list_thumbnail(list_id, req.thumbnail_source_id)

    next_row = await get_list_with_display(list_id)
    if next_row is None:
        raise HTTPException(status_code=404, detail="List not found")
    return await _build_list_response(next_row, request)


@router.delete("/lists/{list_id}", status_code=204)
async def delete_list(list_id: str, cfg: BibilabConfig = Depends(get_config)) -> None:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    placeholders = ",".join("?" * len(_ACTIVE_JOB_STATUSES))
    async with get_db() as db:
        cursor = await db.execute_fetchall(
            f"""
            SELECT 1 FROM jobs
            WHERE json_extract(meta, '$.list_id') = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            (list_id, *_ACTIVE_JOB_STATUSES),
        )
        if len(cursor) > 0:
            raise HTTPException(status_code=409, detail="Cannot delete a list with active jobs")

    artifacts = await get_artifacts_for_list(list_id)
    for artifact in artifacts:
        if artifact["content_path"]:
            (bibilab_home() / artifact["content_path"]).unlink(missing_ok=True)
    await delete_artifacts_for_list(list_id)

    sources = await get_sources_for_list(list_id)
    for source in sources:
        _purge_source_resources(source["id"])

    await delete_sources_for_list(list_id)
    await asyncio.to_thread(clear_embeddings_for_list, list_id, cfg)
    await clear_fts_for_list(list_id)
    await db_delete_list(list_id)


@router.get("/lists/{list_id}/sources")
async def get_list_sources(list_id: str) -> list[SourceResponse]:
    if await get_list(list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")
    rows = await get_sources_for_list(list_id)
    return [
        SourceResponse(
            id=r["id"],
            video_id=r["video_id"],
            platform=r["platform"],
            title=r["title"],
            summary=r["summary"],
            keywords=json.loads(r["keywords"] or "[]"),
            cover_url=r["cover_url"],
            source_url=r["source_url"],
            duration_seconds=r["duration_seconds"],
            uploader=r["uploader"],
            language=r["language"],
            processed_at=r["processed_at"] or "",
        )
        for r in rows
    ]


@router.delete("/lists/{list_id}/sources/{source_id}", status_code=204)
async def delete_list_source(list_id: str, source_id: str, cfg: BibilabConfig = Depends(get_config)) -> None:
    source = await get_source(source_id)
    if source is None or source["list_id"] != list_id:
        raise HTTPException(status_code=404, detail="Source not found")

    placeholders = ",".join("?" * len(_ACTIVE_JOB_STATUSES))
    async with get_db() as db:
        cursor = await db.execute_fetchall(
            f"""
            SELECT 1 FROM jobs
            WHERE json_extract(meta, '$.video_id') = ?
              AND json_extract(meta, '$.list_id') = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            (source["video_id"], list_id, *_ACTIVE_JOB_STATUSES),
        )
        if len(cursor) > 0:
            raise HTTPException(status_code=409, detail="Cannot delete a source with active jobs")

    row = await get_list(list_id)
    if row is not None and row["thumbnail_source_id"] == source_id:
        await update_list_thumbnail(list_id, None)

    _purge_source_resources(source_id)
    await asyncio.to_thread(clear_embeddings_for_source, source_id, cfg)
    await asyncio.to_thread(clear_fts_for_source_sync, source_id)
    await delete_source(source_id)


@router.post("/lists/{list_id}/overview")
async def generate_list_overview(
    list_id: str, request: Request, cfg: BibilabConfig = Depends(get_config)
) -> OverviewResponse:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    sources = await get_sources_for_list(list_id)
    if not sources:
        raise HTTPException(status_code=422, detail="List has no sources to summarise")

    from bibilab.pipeline.extract import generate_overview

    list_videos = [{"title": s["title"], "summary": s["summary"]} for s in sources]
    outline = await asyncio.to_thread(
        generate_overview,
        list_videos,
        cfg.ai,
        cfg.ai.output_language,
        request.headers.get("X-UI-Lang"),
        llm_timeout=cfg.transcription.llm_timeout,
    )

    list_name = row["name"]
    source_lines = "\n".join(f"- {s['title']}" for s in sources)
    content = f"# {list_name} - Overview\n\n## Outline\n{outline}\n\n## Sources\n{source_lines}\n"
    filename = f"overview-{list_name.lower().replace(' ', '-')}.md"

    return OverviewResponse(content=content, filename=filename)
