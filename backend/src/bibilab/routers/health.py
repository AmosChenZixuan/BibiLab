import shutil

import httpx
from fastapi import APIRouter, Depends

from bibilab.asr_models import (
    diarization_model_path,
    is_diarization_model_downloaded,
    is_model_downloaded,
    resolve_model_path,
)
from bibilab.config import BibilabConfig, get_config
from bibilab.pipeline.embed import _embedding_model_dir, is_embedding_model_downloaded
from bibilab.pipeline.rerank import _model_dir, is_reranker_model_downloaded

router = APIRouter()


async def _check_llm(cfg: BibilabConfig) -> dict:
    api_key = cfg.ai.api_key
    base_url = (cfg.ai.base_url or "").strip()

    if not base_url:
        return {"status": "error", "message": "base_url not configured"}

    hosted = base_url in ("https://api.openai.com/v1", "https://api.anthropic.com/v1")
    if not api_key and hosted:
        return {"status": "error", "message": "api_key not configured"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(base_url.rstrip("/"), follow_redirects=True)

        if resp.status_code >= 500:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}

        return {"status": "ok", "message": base_url}
    except httpx.TimeoutException as exc:
        return {"status": "error", "message": f"Request timed out: {exc}"}
    except (httpx.NetworkError, httpx.ProtocolError, httpx.HTTPError) as exc:
        return {"status": "error", "message": f"HTTP error: {exc}"}
    except OSError as exc:
        return {"status": "error", "message": f"Network error: {exc}"}


def _check_asr(cfg: BibilabConfig) -> dict:
    model = cfg.transcription.model
    if not is_model_downloaded(model):
        return {"status": "error", "message": f"Model {model!r} not downloaded"}
    return {"status": "ok", "message": str(resolve_model_path(model))}


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
    if is_diarization_model_downloaded():
        return {"status": "ok", "message": str(diarization_model_path())}
    return {
        "status": "error",
        "message": "Diarization model (CAM++, ~28 MB) not found. Auto-downloads on first ingest.",
    }


def _check_embedding_model() -> dict:
    if is_embedding_model_downloaded():
        return {"status": "ok", "message": str(_embedding_model_dir() / "onnx" / "model.onnx")}
    return {
        "status": "error",
        "message": (
            f"Embedding model not found at {_embedding_model_dir() / 'onnx' / 'model.onnx'}. "
            "It downloads automatically on the first pipeline run (~50 MB)."
        ),
    }


def _check_reranker_model() -> dict:
    if is_reranker_model_downloaded():
        return {"status": "ok", "message": str(_model_dir())}
    return {
        "status": "error",
        "message": (
            f"Reranker model not found at {_model_dir()}. It downloads automatically on first chat query (~140 MB)."
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
