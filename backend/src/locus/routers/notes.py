from pathlib import Path

from fastapi import APIRouter, HTTPException

from locus.db import get_source
from locus.models.notes import NoteContentResponse, NoteTranscriptResponse

router = APIRouter()


@router.get("/notes/{video_id}/content")
async def get_note_content(video_id: str) -> NoteContentResponse:
    source = await get_source(video_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Note not found")
    path = Path(source["note_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Note file not found on disk")
    return NoteContentResponse(
        video_id=video_id,
        title=source["title"],
        markdown=path.read_text(encoding="utf-8"),
    )


@router.get("/notes/{video_id}/transcript")
async def get_note_transcript(video_id: str) -> NoteTranscriptResponse:
    source = await get_source(video_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Note not found")
    if not source["transcript_path"]:
        raise HTTPException(status_code=404, detail="No transcript for this note")
    path = Path(source["transcript_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found on disk")
    return NoteTranscriptResponse(
        video_id=video_id,
        text=path.read_text(encoding="utf-8"),
    )
