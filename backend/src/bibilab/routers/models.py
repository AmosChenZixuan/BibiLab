"""Model registry API — unified listing + download for all local model deps."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from bibilab.config import BibilabConfig, get_config
from bibilab.db.jobs import create_job
from bibilab.model_registry import (
    _integrity_ok,
    _target_dir,
    get_spec,
    list_specs,
    missing_required_models,
    required_models,
)
from bibilab.models.models import ModelDownloadResponse, ModelInfo, SyncResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(cfg: BibilabConfig = Depends(get_config)) -> list[ModelInfo]:
    required_ids = {s.id for s in required_models(cfg)}
    out: list[ModelInfo] = []
    for spec in list_specs():
        installed = _integrity_ok(spec)
        out.append(
            ModelInfo(
                id=spec.id,
                display_name=spec.display_name,
                kind=spec.kind,
                size_mb=spec.size_mb,
                status="present" if installed else "missing",
                required_by_config=spec.id in required_ids,
                path=str(_target_dir(spec)) if installed else None,
            )
        )
    return out


@router.post("/models/{spec_id}/download", status_code=status.HTTP_202_ACCEPTED)
async def download_model(spec_id: str) -> ModelDownloadResponse:
    try:
        get_spec(spec_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = await create_job(
        type="model_download",
        meta={"model_name": spec_id},
    )
    return ModelDownloadResponse(job_id=job_id, status="queued", spec_id=spec_id)


@router.post("/models/sync", status_code=status.HTTP_202_ACCEPTED)
async def sync_models(cfg: BibilabConfig = Depends(get_config)) -> SyncResponse:
    missing = missing_required_models(cfg)
    synced: list[str] = []
    skipped: list[str] = []
    job_ids: list[str] = []

    for spec in required_models(cfg):
        if spec.id in missing:
            try:
                job_id = await create_job(type="model_download", meta={"model_name": spec.id})
                job_ids.append(job_id)
                synced.append(spec.id)
            except Exception:
                logger.exception("Failed to create download job for %s", spec.id)
        else:
            skipped.append(spec.id)

    return SyncResponse(job_ids=job_ids, synced=synced, skipped=skipped)
