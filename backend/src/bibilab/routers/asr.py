from fastapi import APIRouter, Depends, HTTPException, status

from bibilab.asr_models import (
    DIARIZATION_MODELS,
    SUPPORTED_MODELS,
    is_diarization_model_downloaded,
    is_model_downloaded,
    resolve_model_path,
)
from bibilab.config import BibilabConfig, get_config
from bibilab.db import create_job
from bibilab.models.asr import (
    AsrModelDownloadRequest,
    AsrModelDownloadResponse,
    AsrModelInfo,
)

router = APIRouter()


@router.get("/models/asr")
async def list_asr_models(cfg: BibilabConfig = Depends(get_config)) -> list[AsrModelInfo]:
    selected_engine = cfg.transcription.engine
    selected_model = cfg.transcription.model_size
    models: list[AsrModelInfo] = []

    for engine, model_names in SUPPORTED_MODELS.items():
        for name in model_names:
            models.append(
                AsrModelInfo(
                    name=name,
                    engine=engine,
                    installed=is_model_downloaded(engine, name),
                    path=str(resolve_model_path(engine, name)) if resolve_model_path(engine, name) else None,
                    selected=(engine == selected_engine and name == selected_model),
                )
            )

    for name in DIARIZATION_MODELS:
        models.append(
            AsrModelInfo(
                name=name,
                engine="diarization",
                installed=is_diarization_model_downloaded(),
                path=None,
                selected=False,
            )
        )

    return models


@router.post(
    "/models/asr/download",
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_asr_model(req: AsrModelDownloadRequest) -> AsrModelDownloadResponse:
    if req.engine == "diarization":
        if req.model_size not in DIARIZATION_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported diarization model {req.model_size!r}.",
            )
    else:
        if req.engine not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported engine {req.engine!r}.",
            )
        if req.model_size not in SUPPORTED_MODELS[req.engine]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model {req.model_size!r} for engine {req.engine!r}.",
            )

    job_id = await create_job(
        type="model_download",
        meta={
            "engine": req.engine,
            "model_size": req.model_size,
        },
    )
    return AsrModelDownloadResponse(
        job_id=job_id,
        status="queued",
        engine=req.engine,
        model_size=req.model_size,
    )
