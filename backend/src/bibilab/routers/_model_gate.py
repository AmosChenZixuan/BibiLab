"""Pre-flight 412 gate shared by ingest/chat/artifacts routers."""

from fastapi import HTTPException

from bibilab.config import BibilabConfig
from bibilab.model_registry import missing_required_models

MODELS_MISSING_ERROR = "models_missing"


def require_models_present(cfg: BibilabConfig) -> None:
    missing = missing_required_models(cfg)
    if missing:
        raise HTTPException(
            status_code=412,
            detail={"error": MODELS_MISSING_ERROR, "missing": missing},
        )
