"""ASR model registry.

Each model self-identifies its kind (transcription, diarization, or vad).
All models load through FunASR's AutoModel — SenseVoice natively, Whisper
via WhisperWarp (which wraps openai-whisper internally).
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

from bibilab.config import AsrModelKind, models_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    kind: AsrModelKind
    size_mb: int
    modelscope_id: str | None = None  # unset for openai-backed models (large-v3)


_SPECS: dict[str, ModelSpec] = {
    "large-v3": ModelSpec(
        name="large-v3",
        display_name="Faster Whisper large-v3",
        kind="transcription",
        size_mb=3000,
    ),
    "sensevoice-small": ModelSpec(
        name="sensevoice-small",
        display_name="SenseVoice Small",
        kind="transcription",
        size_mb=936,
        modelscope_id="iic/SenseVoiceSmall",
    ),
    "cam++": ModelSpec(
        name="cam++",
        display_name="CAM++ (Speaker Diarization)",
        kind="diarization",
        size_mb=28,
        modelscope_id="iic/speech_campplus_sv_zh-cn_16k-common",
    ),
    "fsmn-vad": ModelSpec(
        name="fsmn-vad",
        display_name="FSMN-VAD (Voice Activity Detection)",
        kind="vad",
        size_mb=4,
        modelscope_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
}

DIARIZATION_MODEL = "cam++"
VAD_MODEL = "fsmn-vad"


def list_specs() -> list[ModelSpec]:
    return list(_SPECS.values())


def get_spec(name: str) -> ModelSpec:
    if name not in _SPECS:
        raise ValueError(f"Unknown ASR model {name!r}")
    return _SPECS[name]


def _target_dir(name: str) -> Path:
    return models_dir("asr", name)


def _whisper_cache_path() -> Path:
    """openai-whisper cache location (honours XDG_CACHE_HOME).

    FunASR's WhisperWarp delegates to openai-whisper internally, which
    downloads to this path. We check it for the UI's installed-status
    without importing openai-whisper ourselves.
    """
    cache_root = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(cache_root) / "whisper" / "large-v3.pt"


def _is_downloaded(spec: ModelSpec) -> bool:
    if spec.name == "large-v3":
        return _whisper_cache_path().exists()
    target = _target_dir(spec.name)
    return target.exists() and (target / "configuration.json").exists()


def _download_modelscope(model_id: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    logger.info("Downloading ASR model from ModelScope: %s", model_id)
    try:
        snapshot_download(model_id, local_dir=str(tmp))
    except Exception:
        logger.exception("ModelScope download failed for %s", model_id)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    logger.info("ASR model downloaded to %s", target)
    return target


def resolve_model_path(name: str) -> Path | None:
    spec = get_spec(name)
    if not _is_downloaded(spec):
        return None
    if spec.name == "large-v3":
        return _whisper_cache_path()
    return _target_dir(name)


def is_model_downloaded(name: str) -> bool:
    return _is_downloaded(get_spec(name))


def is_diarization_model_downloaded() -> bool:
    return is_model_downloaded(DIARIZATION_MODEL)


def diarization_model_path() -> Path | None:
    return resolve_model_path(DIARIZATION_MODEL)


def download_model(name: str) -> Path:
    spec = get_spec(name)
    if spec.name == "large-v3":
        # WhisperWarp downloads via openai-whisper internally on first
        # AutoModel init. Trigger the download through FunASR so we
        # don't import openai-whisper ourselves.
        from funasr import AutoModel  # noqa: PLC0415

        logger.info("Downloading Whisper large-v3 (~3 GB) via WhisperWarp — this may take several minutes")
        try:
            AutoModel(
                model="Whisper-large-v3",
                hub="openai",
                device="cpu",
                disable_update=True,
                disable_pbar=True,
            )
        except Exception:
            logger.exception("Whisper large-v3 download failed")
            raise
        logger.info("Whisper large-v3 download complete → %s", _whisper_cache_path())
        return _whisper_cache_path()
    if spec.modelscope_id is None:
        raise ValueError(f"Model {name!r} has no modelscope_id")
    return _download_modelscope(spec.modelscope_id, _target_dir(name))


__all__ = [
    "DIARIZATION_MODEL",
    "VAD_MODEL",
    "ModelSpec",
    "download_model",
    "get_spec",
    "is_diarization_model_downloaded",
    "is_model_downloaded",
    "list_specs",
    "resolve_model_path",
]
