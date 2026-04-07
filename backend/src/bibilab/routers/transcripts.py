from fastapi import APIRouter, HTTPException, Query

from bibilab.config import transcript_path
from bibilab.models.transcripts import TranscriptResponse

router = APIRouter()


@router.get("/transcripts/{video_id}")
async def get_transcript(
    video_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1),
) -> TranscriptResponse:
    _transcript_path = transcript_path(video_id)
    if not _transcript_path.exists() or not _transcript_path.is_file():
        raise HTTPException(status_code=404, detail="Transcript not found")

    lines = _transcript_path.read_text(encoding="utf-8").splitlines()
    return TranscriptResponse(
        video_id=video_id,
        total_lines=len(lines),
        offset=offset,
        lines=lines[offset : offset + limit],
    )
