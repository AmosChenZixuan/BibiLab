from fastapi import APIRouter, HTTPException, status

from bibilab.config import load_config
from bibilab.db import create_job
from bibilab.models.whisper import (
    WhisperModelDownloadRequest,
    WhisperModelInfo,
)
from bibilab.whisper_models import (
    SUPPORTED_WHISPER_MODELS,
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
            path=(str(resolve_local_model_path(name)) if resolve_local_model_path(name) is not None else None),
            selected=name == selected_model,
        )
        for name in SUPPORTED_WHISPER_MODELS
    ]


@router.post(
    "/models/whisper/download",
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_whisper(req: WhisperModelDownloadRequest) -> dict:
    if req.model_size not in SUPPORTED_WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(f"Unsupported whisper model {req.model_size!r}. Supported: {', '.join(SUPPORTED_WHISPER_MODELS)}"),
        )

    job_id = await create_job(
        type="model_download",
        meta={
            "model_family": "whisper",
            "model_size": req.model_size,
        },
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model_family": "whisper",
        "model_size": req.model_size,
    }
