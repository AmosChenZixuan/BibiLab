import shutil

import httpx
from fastapi import APIRouter

from bibilab.config import load_config
from bibilab.pipeline.embed import _embedding_model_dir, is_embedding_model_downloaded
from bibilab.whisper_models import is_whisper_model_downloaded, whisper_model_dir

router = APIRouter()


async def _check_llm(cfg) -> dict:
    provider = cfg.ai.provider
    api_key = cfg.ai.api_key
    base_url = (cfg.ai.base_url or "").strip()

    if not base_url:
        return {"status": "error", "message": "base_url not configured"}

    if not api_key and provider not in ("ollama", "custom"):
        return {"status": "error", "message": "api_key not configured"}

    try:
        if provider == "openai":
            url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider == "anthropic":
            url = f"{base_url.rstrip('/')}/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        elif provider == "ollama":
            url = f"{base_url.rstrip('/')}/api/tags"
            headers = {}
        elif provider == "custom":
            url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        else:
            return {"status": "error", "message": f"Unknown provider: {provider}"}

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code >= 400:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}

        payload = resp.json()
        if provider == "ollama":
            if not isinstance(payload.get("models"), list):
                return {"status": "error", "message": "Invalid tags response"}
        elif not isinstance(payload.get("data"), list):
            return {"status": "error", "message": "Invalid models response"}

        return {"status": "ok", "message": base_url}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _check_whisper(cfg) -> dict:
    model_size = cfg.transcription.model_size
    if not is_whisper_model_downloaded(model_size):
        return {
            "status": "error",
            "message": (f"Model {model_size!r} not downloaded to {whisper_model_dir()}"),
        }
    return {"status": "ok", "message": ""}


def _check_ffmpeg() -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return {"status": "ok", "message": ffmpeg_path}
    return {"status": "error", "message": "ffmpeg not found on PATH"}


def _check_cuda() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {"status": "ok", "message": name}
        return {
            "status": "unavailable",
            "message": "CUDA not available; CPU will be used",
        }
    except ImportError:
        return {
            "status": "unavailable",
            "message": "torch not installed; CPU will be used",
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


@router.get("/health")
async def health() -> dict:
    cfg = load_config()

    deps = {
        "backend": {"status": "ok", "message": ""},
        "llm": await _check_llm(cfg),
        "whisper_model": _check_whisper(cfg),
        "ffmpeg": _check_ffmpeg(),
        "cuda": _check_cuda(),
        "embedding_model": _check_embedding_model(),
    }

    blocking = {"llm", "whisper_model", "ffmpeg"}
    has_error = any(v["status"] == "error" for k, v in deps.items() if k in blocking)
    overall = "error" if has_error else "ok"

    return {"overall": overall, "dependencies": deps}
