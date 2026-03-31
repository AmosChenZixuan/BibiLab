import asyncio

from fastapi import APIRouter, HTTPException, status

from locus.config import load_config
from locus.models.whisper import (
    WhisperModelDownloadRequest,
    WhisperModelDownloadResponse,
    WhisperModelInfo,
)
from locus.whisper_models import (
    SUPPORTED_WHISPER_MODELS,
    download_whisper_model,
    is_whisper_model_downloaded,
    resolve_local_model_path,
)

router = APIRouter()


@router.get("/models/whisper")
async def list_whisper_models() -> list[WhisperModelInfo]:
    selected_model = load_config().transcription.model_size
    return [
        WhisperModelInfo(
            name=name,
            installed=is_whisper_model_downloaded(name),
            path=(
                str(resolve_local_model_path(name))
                if resolve_local_model_path(name) is not None
                else None
            ),
            selected=name == selected_model,
        )
        for name in SUPPORTED_WHISPER_MODELS
    ]


@router.post(
    "/models/whisper/download",
    status_code=status.HTTP_201_CREATED,
)
async def download_whisper(req: WhisperModelDownloadRequest) -> WhisperModelDownloadResponse:
    if req.model_size not in SUPPORTED_WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported whisper model {req.model_size!r}. "
                f"Supported: {', '.join(SUPPORTED_WHISPER_MODELS)}"
            ),
        )

    try:
        path = await asyncio.to_thread(download_whisper_model, req.model_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    selected_model = load_config().transcription.model_size
    return WhisperModelDownloadResponse(
        name=req.model_size,
        path=str(path),
        selected=req.model_size == selected_model,
    )
