import shutil

import httpx
from fastapi import APIRouter

from locus.config import load_config

router = APIRouter()


async def _check_llm(cfg) -> dict:
    provider = cfg.ai.provider
    api_key = cfg.ai.api_key
    base_url = cfg.ai.base_url

    if not api_key and provider not in ("ollama", "custom"):
        return {"status": "error", "message": "api_key not configured"}

    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        elif provider in ("ollama", "custom"):
            root = base_url or "http://localhost:11434"
            url = f"{root.rstrip('/')}/api/tags"
            headers = {}
        else:
            return {"status": "error", "message": f"Unknown provider: {provider}"}

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code < 400:
            return {"status": "ok", "message": ""}
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _check_whisper(cfg) -> dict:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return {"status": "error", "message": "huggingface_hub not installed"}

    model_size = cfg.transcription.model_size
    # faster-whisper uses Systran/faster-whisper-{size} on HuggingFace
    repo_id = f"Systran/faster-whisper-{model_size}"
    result = try_to_load_from_cache(repo_id, "config.json")
    if result is None:
        return {"status": "error", "message": f"Model {model_size!r} not downloaded"}
    return {"status": "ok", "message": ""}


def _check_ffmpeg() -> dict:
    if shutil.which("ffmpeg"):
        return {"status": "ok", "message": ""}
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


def _check_bilibili(cfg) -> dict:
    if cfg.accounts.bilibili.cookie:
        return {"status": "ok", "message": "Cookie configured (not validated)"}
    return {"status": "error", "message": "Bilibili cookie not configured"}


@router.get("/health")
async def health() -> dict:
    cfg = load_config()

    deps = {
        "backend": {"status": "ok", "message": ""},
        "llm": await _check_llm(cfg),
        "whisper_model": _check_whisper(cfg),
        "ffmpeg": _check_ffmpeg(),
        "cuda": _check_cuda(),
        "bilibili_session": _check_bilibili(cfg),
    }

    blocking = {"llm", "whisper_model", "ffmpeg"}
    has_error = any(v["status"] == "error" for k, v in deps.items() if k in blocking)
    overall = "error" if has_error else "ok"

    return {"overall": overall, "dependencies": deps}
