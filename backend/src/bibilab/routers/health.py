import shutil

from fastapi import APIRouter, Depends

from bibilab.config import BibilabConfig, get_config
from bibilab.model_registry import (
    DIARIZATION_SPEC_ID,
    EMBEDDING_SPEC_ID,
    RERANKER_SPEC_ID,
    _integrity_ok,
    _target_dir,
    get_spec,
)

router = APIRouter()


async def _check_llm(cfg: BibilabConfig) -> dict:
    base_url = (cfg.ai.base_url or "").strip()
    model = (cfg.ai.model or "").strip()
    if not base_url:
        return {"status": "error", "message": "base_url not configured"}
    if not model:
        return {"status": "error", "message": "model not configured"}
    return {"status": "configured", "message": base_url}


def _check_asr(cfg: BibilabConfig) -> dict:
    model = cfg.transcription.model
    if not model:
        return {"status": "error", "message": "Transcription model not configured"}
    try:
        spec = get_spec(model)
    except ValueError:
        return {"status": "error", "message": f"Unknown transcription model {model!r}"}
    if not _integrity_ok(spec):
        return {"status": "error", "message": f"Model {model!r} not downloaded"}
    return {"status": "ok", "message": str(_target_dir(spec))}


def _check_ffmpeg() -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return {"status": "ok", "message": ffmpeg_path}
    return {"status": "error", "message": "ffmpeg not found on PATH"}


def _check_cuda() -> dict:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return {"status": "ok", "message": f"CUDA available ({torch.cuda.get_device_name(0)})"}
        return {"status": "unavailable", "message": "CUDA not available on this device"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "message": f"CUDA probe failed: {exc}"}


def _check_diarization_model() -> dict:
    spec = get_spec(DIARIZATION_SPEC_ID)
    if _integrity_ok(spec):
        return {"status": "ok", "message": str(_target_dir(spec))}
    return {
        "status": "error",
        "message": "Diarization model (CAM++, ~28 MB) not found. Auto-downloads on first ingest.",
    }


def _check_embedding_model() -> dict:
    spec = get_spec(EMBEDDING_SPEC_ID)
    if _integrity_ok(spec):
        return {"status": "ok", "message": str(_target_dir(spec))}
    return {
        "status": "error",
        "message": (
            f"Embedding model not found at {_target_dir(spec)}. "
            "It downloads automatically on the first pipeline run (~50 MB)."
        ),
    }


def _check_reranker_model() -> dict:
    spec = get_spec(RERANKER_SPEC_ID)
    if _integrity_ok(spec):
        return {"status": "ok", "message": str(_target_dir(spec))}
    return {
        "status": "error",
        "message": (
            f"Reranker model not found at {_target_dir(spec)}. "
            f"It downloads automatically on first chat query (~{spec.size_mb} MB)."
        ),
    }


@router.get("/health")
async def health(cfg: BibilabConfig = Depends(get_config)) -> dict:

    deps = {
        "backend": {"status": "ok", "message": ""},
        "llm": await _check_llm(cfg),
        "asr_model": _check_asr(cfg),
        "ffmpeg": _check_ffmpeg(),
        "cuda": _check_cuda(),
        "embedding_model": _check_embedding_model(),
        "reranker_model": _check_reranker_model(),
        "diarization_model": _check_diarization_model(),
    }

    blocking = {"llm", "asr_model", "ffmpeg"}
    has_error = any(v["status"] == "error" for k, v in deps.items() if k in blocking)
    overall = "error" if has_error else "ok"

    return {"overall": overall, "dependencies": deps}
