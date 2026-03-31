from fastapi import APIRouter, HTTPException, Response, status

from locus.db import get_db
from locus.models.notes import NotePathUpdateRequest

router = APIRouter()


@router.patch("/notes/{locus_id}/path", status_code=status.HTTP_204_NO_CONTENT)
async def update_note_path(locus_id: str, req: NotePathUpdateRequest) -> Response:
    async with get_db() as db:
        async with db.execute(
            "SELECT 1 FROM processing_log WHERE video_id=?",
            (locus_id,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Note not found")

        await db.execute(
            "UPDATE processing_log SET note_path=? WHERE video_id=?",
            (req.path, locus_id),
        )
        await db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
