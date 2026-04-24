import shutil

import httpx
from fastapi import APIRouter, Depends

from bibilab.config import BibilabConfig, get_config
from bibilab.pipeline.embed import _embedding_model_dir, is_embedding_model_downloaded
from bibilab.whisper_models import is_whisper_model_downloaded, whisper_model_dir

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


def _check_whisper(cfg: BibilabConfig) -> dict:
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
        import ctypes
        from pathlib import Path

        try:
            import nvidia.cublas

            lib_path = Path(nvidia.cublas.__path__[0]) / "lib" / "libcublas.so.12"
            ctypes.CDLL(str(lib_path))
        except ImportError:
            ctypes.CDLL("libcublas.so.12")
        return {"status": "ok", "message": "CUDA available"}
    except OSError as exc:
        return {
            "status": "unavailable",
            "message": (f"CUDA libraries not available: {exc}. Install with: uv sync --extra cuda"),
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
async def health(cfg: BibilabConfig = Depends(get_config)) -> dict:

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
