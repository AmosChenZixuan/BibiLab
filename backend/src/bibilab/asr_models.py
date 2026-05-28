"""Unified ASR model registry — Whisper + SenseVoice + diarization."""

import shutil
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

from bibilab.config import SUPPORTED_MODELS, AsrModelKind, bibilab_home, models_dir

SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"
CAMPP_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def _hf_cache_dir() -> Path:
    return bibilab_home() / ".cache" / "huggingface"


def _has_funasr_files(path: Path) -> bool:
    return (path / "model.pt").exists() and (path / "config.yaml").exists()


def _atomic_snapshot_download(model_id: str, target: Path) -> Path:
    """Download to a sibling tmp dir, then rename. Leaves no half-populated target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        snapshot_download(model_id, local_dir=str(tmp))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


# ── Whisper ───────────────────────────────────────────────────────────────


def _whisper_target_dir(model_size: str) -> Path:
    return models_dir("whisper") / model_size


def _resolve_whisper_path(model_size: str) -> Path | None:
    path = _whisper_target_dir(model_size)
    if (path / "config.json").exists() and (path / "model.bin").exists():
        return path
    return None


def _is_whisper_downloaded(model_size: str) -> bool:
    return _resolve_whisper_path(model_size) is not None


def _download_whisper(model_size: str) -> Path:
    from faster_whisper.utils import download_model as fw_download

    target = _whisper_target_dir(model_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = _hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{model_size}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        fw_download(model_size, output_dir=str(tmp), cache_dir=str(cache))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(tmp / ".cache", ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


# ── SenseVoice ────────────────────────────────────────────────────────────


def _sensevoice_target_dir() -> Path:
    return models_dir("sensevoice") / "small"


def _is_sensevoice_downloaded() -> bool:
    return _has_funasr_files(_sensevoice_target_dir())


def _download_sensevoice() -> Path:
    return _atomic_snapshot_download(SENSEVOICE_MODEL_ID, _sensevoice_target_dir())


# ── Diarization (CAM++) ───────────────────────────────────────────────────


def _diarization_target_dir() -> Path:
    return models_dir("diarization") / "cam++"


def _is_diarization_downloaded() -> bool:
    return _has_funasr_files(_diarization_target_dir())


def _download_diarization() -> Path:
    return _atomic_snapshot_download(CAMPP_MODEL_ID, _diarization_target_dir())


# ── Public API ────────────────────────────────────────────────────────────


def resolve_model_path(engine: str, model_size: str) -> Path | None:
    if engine == "whisper":
        return _resolve_whisper_path(model_size)
    if engine == "sensevoice":
        return _sensevoice_target_dir() if _is_sensevoice_downloaded() else None
    if engine == "diarization":
        return _diarization_target_dir() if _is_diarization_downloaded() else None
    return None


def is_model_downloaded(engine: str, model_size: str) -> bool:
    if engine == "whisper":
        return _is_whisper_downloaded(model_size)
    if engine == "sensevoice":
        return _is_sensevoice_downloaded()
    if engine == "diarization":
        return _is_diarization_downloaded()
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
    if engine == "diarization":
        return _download_diarization()
    raise ValueError(f"Unknown engine: {engine!r}")


def download_diarization_model() -> Path:
    return _download_diarization()


__all__ = [
    "AsrModelKind",
    "CAMPP_MODEL_ID",
    "SENSEVOICE_MODEL_ID",
    "VAD_MODEL_ID",
    "download_diarization_model",
    "download_model",
    "is_diarization_model_downloaded",
    "is_model_downloaded",
    "resolve_model_path",
]
