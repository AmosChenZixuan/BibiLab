from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bibilab.config import resolve_storage_path
from bibilab.db import get_source
from bibilab.models.notes import NoteContentResponse, NoteTranscriptResponse

router = APIRouter()


@router.get("/notes/{video_id}/content")
async def get_note_content(video_id: str) -> NoteContentResponse:
    source = await get_source(video_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Note not found")
    path = resolve_storage_path(source["note_path"])
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
    path = resolve_storage_path(source["transcript_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found on disk")
    return NoteTranscriptResponse(
        video_id=video_id,
        text=path.read_text(encoding="utf-8"),
    )


@router.get("/notes/{video_id}/attachments/{path:path}")
async def get_note_attachment(video_id: str, path: str):
    """Serve files from the note's attachments directory (e.g. images referenced in markdown)."""
    source = await get_source(video_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Note not found")
    note_path = resolve_storage_path(source["note_path"])
    # attachments/ is a sibling of the note file
    attachment_path = note_path.parent / "attachments" / path
    if not attachment_path.exists() or not attachment_path.is_file():
        raise HTTPException(status_code=404, detail="Attachment not found")
    return FileResponse(attachment_path)
