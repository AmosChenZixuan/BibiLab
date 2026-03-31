import asyncio
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from locus.config import load_config
from locus.db import get_db
from locus.models.lists import ListCreateRequest, ListNoteResponse, ListResponse
from locus.pipeline.embed import clear_embeddings_for_list
from locus.vault import get_list_by_id, locus_dir, scan_lists, write_list_overview

router = APIRouter()

_ACTIVE_JOB_STATUSES = ("queued", "downloading", "transcribing", "extracting", "writing")


@router.post("/lists", status_code=201)
async def create_list(req: ListCreateRequest) -> ListResponse:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="List name cannot be empty")

    cfg = load_config()
    if not cfg.obsidian.vault_path:
        raise HTTPException(
            status_code=400,
            detail="obsidian.vault_path is not configured. Set it via PUT /config.",
        )

    list_id = str(uuid.uuid4())
    list_dir = locus_dir(cfg.obsidian) / name
    try:
        list_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=f"List {name!r} already exists") from exc

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        await asyncio.to_thread(
            write_list_overview,
            list_dir / "_overview.md",
            list_id=list_id,
            list_name=name,
            created_at=now,
        )
    except Exception:
        await asyncio.to_thread(shutil.rmtree, list_dir, True)
        raise

    return ListResponse(id=list_id, name=name, created_at=now)


@router.get("/lists")
async def get_lists() -> list[ListResponse]:
    cfg = load_config()
    if not cfg.obsidian.vault_path:
        return []

    lists = await asyncio.to_thread(scan_lists, cfg.obsidian)
    return [ListResponse(id=item.id, name=item.name, created_at=item.created_at) for item in lists]


@router.get("/lists/{list_id}/notes")
async def get_list_notes(list_id: str) -> list[ListNoteResponse]:
    cfg = load_config()
    if not cfg.obsidian.vault_path:
        raise HTTPException(status_code=404, detail="List not found")

    list_meta = await asyncio.to_thread(get_list_by_id, list_id, cfg.obsidian)
    if list_meta is None:
        raise HTTPException(status_code=404, detail="List not found")

    async with get_db() as db:
        async with db.execute(
            """
            SELECT video_id, note_path, processed_at, platform
            FROM processing_log
            WHERE list_id=?
            ORDER BY processed_at DESC
            """,
            (list_id,),
        ) as cur:
            rows = await cur.fetchall()

    return [ListNoteResponse.model_validate(dict(row)) for row in rows]


@router.delete("/lists/{list_id}", status_code=204)
async def delete_list(list_id: str) -> None:
    cfg = load_config()
    if not cfg.obsidian.vault_path:
        raise HTTPException(status_code=404, detail="List not found")

    list_meta = await asyncio.to_thread(get_list_by_id, list_id, cfg.obsidian)
    if list_meta is None:
        raise HTTPException(status_code=404, detail="List not found")

    placeholders = ",".join("?" * len(_ACTIVE_JOB_STATUSES))
    async with get_db() as db:
        async with db.execute(
            f"""
            SELECT 1
            FROM jobs
            WHERE json_extract(meta, '$.list_id') = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            (list_id, *_ACTIVE_JOB_STATUSES),
        ) as cur:
            active_job = await cur.fetchone()

        if active_job is not None:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete a list with active jobs",
            )

        await db.execute("DELETE FROM processing_log WHERE list_id=?", (list_id,))
        await db.commit()

    await asyncio.to_thread(clear_embeddings_for_list, list_id, cfg)
    await asyncio.to_thread(shutil.rmtree, list_meta.path, True)
