from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    # Stub — real dependency checks wired in Phase 2
    return {
        "overall": "ok",
        "dependencies": {
            "backend": {"status": "ok", "message": ""},
            "llm": {"status": "unknown", "message": "not yet checked"},
            "whisper_model": {"status": "unknown", "message": "not yet checked"},
            "ffmpeg": {"status": "unknown", "message": "not yet checked"},
            "cuda": {"status": "unknown", "message": "not yet checked"},
            "bilibili_session": {"status": "unknown", "message": "not yet checked"},
        },
    }
