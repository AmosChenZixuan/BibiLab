from fastapi import APIRouter, Depends, HTTPException, status

from bibilab.asr_models import (
    get_spec,
    is_model_downloaded,
    list_specs,
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
    selected = cfg.transcription.model
    out: list[AsrModelInfo] = []
    for spec in list_specs():
        path = resolve_model_path(spec.name)
        out.append(
            AsrModelInfo(
                name=spec.name,
                kind=spec.kind,
                installed=is_model_downloaded(spec.name),
                path=str(path) if path is not None else None,
                selected=(spec.name == selected),
                size_mb=spec.size_mb,
            )
        )
    return out


@router.post("/models/asr/download", status_code=status.HTTP_202_ACCEPTED)
async def download_asr_model(req: AsrModelDownloadRequest) -> AsrModelDownloadResponse:
    try:
        get_spec(req.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job_id = await create_job(
        type="model_download",
        meta={"model_name": req.model_name},
    )
    return AsrModelDownloadResponse(
        job_id=job_id,
        status="queued",
        model_name=req.model_name,
    )
