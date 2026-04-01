import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from locus.config import load_config
from locus.db import (
    create_list as db_create_list,
)
from locus.db import (
    delete_list as db_delete_list,
)
from locus.db import (
    delete_source,
    delete_sources_for_list,
    get_all_lists,
    get_db,
    get_list,
    get_source,
    get_sources_for_list,
)
from locus.models.lists import ListCreateRequest, ListResponse, OverviewResponse, SourceResponse
from locus.pipeline.embed import clear_embeddings_for_list, clear_embeddings_for_video

router = APIRouter()

_ACTIVE_JOB_STATUSES = ("queued", "downloading", "transcribing", "extracting", "writing")


@router.post("/lists", status_code=201)
async def create_list(req: ListCreateRequest) -> ListResponse:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="List name cannot be empty")
    list_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db_create_list(list_id, name, now)
    return ListResponse(id=list_id, name=name, created_at=now)


@router.get("/lists")
async def get_lists() -> list[ListResponse]:
    rows = await get_all_lists()
    return [ListResponse(id=r["id"], name=r["name"], created_at=r["created_at"]) for r in rows]


@router.delete("/lists/{list_id}", status_code=204)
async def delete_list(list_id: str) -> None:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    placeholders = ",".join("?" * len(_ACTIVE_JOB_STATUSES))
    async with get_db() as db:
        async with db.execute(
            f"""
            SELECT 1 FROM jobs
            WHERE json_extract(meta, '$.list_id') = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            (list_id, *_ACTIVE_JOB_STATUSES),
        ) as cur:
            if await cur.fetchone() is not None:
                raise HTTPException(status_code=409, detail="Cannot delete a list with active jobs")

    sources = await get_sources_for_list(list_id)
    for source in sources:
        Path(source["note_path"]).unlink(missing_ok=True)

    await delete_sources_for_list(list_id)
    cfg = load_config()
    await asyncio.to_thread(clear_embeddings_for_list, list_id, cfg)
    await db_delete_list(list_id)


@router.get("/lists/{list_id}/sources")
async def get_list_sources(list_id: str) -> list[SourceResponse]:
    if await get_list(list_id) is None:
        raise HTTPException(status_code=404, detail="List not found")
    rows = await get_sources_for_list(list_id)
    return [
        SourceResponse(
            video_id=r["video_id"],
            platform=r["platform"],
            title=r["title"],
            note_path=r["note_path"],
            processed_at=r["processed_at"] or "",
        )
        for r in rows
    ]


@router.delete("/lists/{list_id}/sources/{video_id}", status_code=204)
async def delete_list_source(list_id: str, video_id: str) -> None:
    source = await get_source(video_id)
    if source is None or source["list_id"] != list_id:
        raise HTTPException(status_code=404, detail="Source not found")

    Path(source["note_path"]).unlink(missing_ok=True)
    cfg = load_config()
    await asyncio.to_thread(clear_embeddings_for_video, video_id, cfg)
    await delete_source(video_id)


@router.post("/lists/{list_id}/overview")
async def generate_list_overview(list_id: str) -> OverviewResponse:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    sources = await get_sources_for_list(list_id)
    if not sources:
        raise HTTPException(status_code=422, detail="List has no sources to summarise")

    cfg = load_config()
    from locus.pipeline.extract import generate_overview

    list_videos = [{"title": s["title"], "summary": s["summary"]} for s in sources]
    outline = await asyncio.to_thread(generate_overview, list_videos, cfg.ai)

    list_name = row["name"]
    source_lines = "\n".join(f"- {s['title']}" for s in sources)
    content = f"# {list_name} - Overview\n\n## Outline\n{outline}\n\n## Sources\n{source_lines}\n"
    filename = f"overview-{list_name.lower().replace(' ', '-')}.md"

    return OverviewResponse(content=content, filename=filename)
