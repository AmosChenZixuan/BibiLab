"""Unified ASR model registry — Whisper + SenseVoice + diarization."""

import shutil
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

from bibilab.config import SUPPORTED_MODELS, bibilab_home, models_dir

# ── Path helpers ──────────────────────────────────────────────────────────


def _whisper_model_dir() -> Path:
    return models_dir("whisper")


def _sensevoice_model_dir() -> Path:
    return models_dir("sensevoice")


def _diarization_model_dir() -> Path:
    return models_dir("diarization")


def _hf_cache_dir() -> Path:
    return bibilab_home() / ".cache" / "huggingface"


# ── Whisper detection / download ──────────────────────────────────────────


def _candidate_whisper_paths(model_size: str) -> list[Path]:
    root = _whisper_model_dir()
    return [
        root / "whisper" / f"whisper-{model_size}",
        root / model_size,
        root / f"faster-whisper-{model_size}",
    ]


def _resolve_whisper_path(model_size: str) -> Path | None:
    for path in _candidate_whisper_paths(model_size):
        if (path / "config.json").exists() and (path / "model.bin").exists():
            return path
    return None


def _is_whisper_downloaded(model_size: str) -> bool:
    if _resolve_whisper_path(model_size) is not None:
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    repo_id = f"Systran/faster-whisper-{model_size}"
    return try_to_load_from_cache(repo_id, "config.json", cache_dir=_hf_cache_dir()) is not None


def _download_whisper(model_size: str) -> Path:
    from faster_whisper.utils import download_model

    model_root = _whisper_model_dir()
    model_root.mkdir(parents=True, exist_ok=True)
    cache_root = _hf_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    target_dir = model_root / model_size
    target_dir.mkdir(parents=True, exist_ok=True)
    download_model(model_size, output_dir=str(target_dir), cache_dir=str(cache_root))
    shutil.rmtree(target_dir / ".cache", ignore_errors=True)
    return _resolve_whisper_path(model_size) or target_dir


# ── SenseVoice detection / download ───────────────────────────────────────

SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"


def _sensevoice_target_dir() -> Path:
    return _sensevoice_model_dir() / "small"


def _is_sensevoice_downloaded() -> bool:
    target = _sensevoice_target_dir()
    return (target / "model.pt").exists() and (target / "config.yaml").exists()


def _download_sensevoice() -> Path:
    target = _sensevoice_target_dir()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(SENSEVOICE_MODEL_ID, local_dir=str(target))
    return target


# ── Diarization (CAM++) detection / download ──────────────────────────────

CAMPP_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"


def _diarization_target_dir() -> Path:
    return _diarization_model_dir() / "cam++"


def _is_diarization_downloaded() -> bool:
    target = _diarization_target_dir()
    return (target / "model.pt").exists() and (target / "config.yaml").exists()


def _download_diarization() -> Path:
    target = _diarization_target_dir()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(CAMPP_MODEL_ID, local_dir=str(target))
    return target


# ── Public API ────────────────────────────────────────────────────────────


def resolve_model_path(engine: str, model_size: str) -> Path | None:
    if engine == "whisper":
        return _resolve_whisper_path(model_size)
    if engine == "sensevoice":
        return _sensevoice_target_dir() if _is_sensevoice_downloaded() else None
    return None


def is_model_downloaded(engine: str, model_size: str) -> bool:
    if engine == "whisper":
        return _is_whisper_downloaded(model_size)
    if engine == "sensevoice":
        return _is_sensevoice_downloaded()
    return False


def is_diarization_model_downloaded() -> bool:
    return _is_diarization_downloaded()


def download_model(engine: str, model_size: str) -> Path:
    if engine not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported engine: {engine!r}")
    if model_size not in SUPPORTED_MODELS[engine]:
        raise ValueError(f"Unsupported model {model_size!r} for engine {engine!r}")

    if engine == "whisper":
        return _download_whisper(model_size)
    if engine == "sensevoice":
        return _download_sensevoice()
    raise ValueError(f"Unknown engine: {engine!r}")


def download_diarization_model() -> Path:
    return _download_diarization()
