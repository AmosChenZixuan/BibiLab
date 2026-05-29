from fastapi import APIRouter, Depends, HTTPException, status

from bibilab.config import BibilabConfig, get_config
from bibilab.db import create_job
from bibilab.model_registry import _integrity_ok, _target_dir, get_spec, list_specs
from bibilab.models.asr import (
    AsrModelDownloadRequest,
    AsrModelDownloadResponse,
    AsrModelInfo,
)

router = APIRouter()


@router.get("/models/asr")
async def list_asr_models(cfg: BibilabConfig = Depends(get_config)) -> list[AsrModelInfo]:
    selected = cfg.transcription.model
    asr_kinds = {"transcription", "diarization", "vad"}
    out: list[AsrModelInfo] = []
    for spec in list_specs():
        if spec.kind not in asr_kinds:
            continue
        installed = _integrity_ok(spec)
        target = _target_dir(spec)
        out.append(
            AsrModelInfo(
                name=spec.id,
                display_name=spec.display_name,
                kind=spec.kind,  # type: ignore[arg-type]
                installed=installed,
                path=str(target) if installed else None,
                selected=(spec.id == selected),
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
