import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from locus.config import load_config
from locus.db import get_db
from locus.models.lists import ListCreateRequest, ListNoteResponse, ListResponse
from locus.pipeline.embed import clear_embeddings_for_list

router = APIRouter()

_ACTIVE_JOB_STATUSES = ("queued", "downloading", "transcribing", "extracting", "writing")


async def _get_list_row(list_id: str):
    async with get_db() as db:
        async with db.execute("SELECT * FROM lists WHERE id=?", (list_id,)) as cur:
            return await cur.fetchone()


@router.post("/lists", status_code=201)
async def create_list(req: ListCreateRequest) -> ListResponse:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="List name cannot be empty")

    list_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    async with get_db() as db:
        try:
            await db.execute(
                "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
                (list_id, name, created_at),
            )
            await db.commit()
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise HTTPException(
                    status_code=409, detail=f"List {name!r} already exists"
                ) from exc
            raise

    return ListResponse(id=list_id, name=name, created_at=created_at)


@router.get("/lists")
async def get_lists() -> list[ListResponse]:
    async with get_db() as db:
        async with db.execute("SELECT * FROM lists ORDER BY created_at ASC") as cur:
            rows = await cur.fetchall()
    return [ListResponse.model_validate(dict(r)) for r in rows]


@router.get("/lists/{list_id}/notes")
async def get_list_notes(list_id: str) -> list[ListNoteResponse]:
    if await _get_list_row(list_id) is None:
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
    if await _get_list_row(list_id) is None:
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
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()

    cfg = load_config()
    await asyncio.to_thread(clear_embeddings_for_list, list_id, cfg)
