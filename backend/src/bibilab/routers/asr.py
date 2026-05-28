from fastapi import APIRouter, Depends, status

from bibilab.asr_models import is_model_downloaded, resolve_model_path
from bibilab.config import SUPPORTED_MODELS, BibilabConfig, get_config
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
            path = resolve_model_path(engine, name)
            models.append(
                AsrModelInfo(
                    name=name,
                    engine=engine,
                    installed=is_model_downloaded(engine, name),
                    path=str(path) if path is not None else None,
                    selected=(engine == selected_engine and name == selected_model),
                )
            )

    return models


@router.post("/models/asr/download", status_code=status.HTTP_202_ACCEPTED)
async def download_asr_model(req: AsrModelDownloadRequest) -> AsrModelDownloadResponse:
    job_id = await create_job(
        type="model_download",
        meta={"engine": req.engine, "model_size": req.model_size},
    )
    return AsrModelDownloadResponse(
        job_id=job_id,
        status="queued",
        engine=req.engine,
        model_size=req.model_size,
    )
