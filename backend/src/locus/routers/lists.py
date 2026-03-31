import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from locus.db import get_db
from locus.models.lists import ListCreateRequest, ListResponse

router = APIRouter()


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
