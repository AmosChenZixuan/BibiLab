"""ASR model registry.

Each model self-identifies its kind (transcription vs diarization), size, and
loader backend. There is no separate "engine" axis — the model name carries
everything callers need. `large-v3` loads via openai-whisper, the rest via
FunASR / ModelScope.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from modelscope.hub.snapshot_download import snapshot_download

from bibilab.config import AsrModelKind, models_dir

_Backend = Literal["whisper", "funasr"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    kind: AsrModelKind  # "transcription" | "diarization"
    backend: _Backend
    size_mb: int
    modelscope_id: str | None = None  # only set for funasr-backed models


_SPECS: dict[str, ModelSpec] = {
    "large-v3": ModelSpec(
        name="large-v3",
        display_name="Faster Whisper large-v3",
        kind="transcription",
        backend="whisper",
        size_mb=3000,
    ),
    "sensevoice-small": ModelSpec(
        name="sensevoice-small",
        display_name="SenseVoice Small",
        kind="transcription",
        backend="funasr",
        size_mb=936,
        modelscope_id="iic/SenseVoiceSmall",
    ),
    "cam++": ModelSpec(
        name="cam++",
        display_name="CAM++ (Speaker Diarization)",
        kind="diarization",
        backend="funasr",
        size_mb=28,
        modelscope_id="iic/speech_campplus_sv_zh-cn_16k-common",
    ),
    "ct-punc": ModelSpec(
        name="ct-punc",
        display_name="CT-Transformer (Punctuation)",
        kind="punctuation",
        backend="funasr",
        size_mb=1100,
        modelscope_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    ),
}

# Shared VAD used by FunASR when CAM++ is attached as spk_model.
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

DIARIZATION_MODEL = "cam++"
PUNCTUATION_MODEL = "ct-punc"


def list_specs() -> list[ModelSpec]:
    return list(_SPECS.values())


def get_spec(name: str) -> ModelSpec:
    if name not in _SPECS:
        raise ValueError(f"Unknown ASR model {name!r}")
    return _SPECS[name]


def _target_dir(name: str) -> Path:
    return models_dir("asr", name)


def _whisper_checkpoint(name: str) -> Path:
    return _target_dir(name) / f"{name}.pt"


def _is_downloaded(spec: ModelSpec) -> bool:
    target = _target_dir(spec.name)
    if spec.backend == "whisper":
        return _whisper_checkpoint(spec.name).exists()
    return target.exists() and (target / "configuration.json").exists()


def _download_modelscope(model_id: str, target: Path) -> Path:
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


def _download_whisper(name: str) -> Path:
    import whisper  # noqa: PLC0415

    whisper_name = name
    if whisper_name not in whisper._MODELS:
        raise ValueError(f"Unknown Whisper checkpoint {whisper_name!r}")
    target = _target_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    whisper._download(whisper._MODELS[whisper_name], str(target), False)
    return target


def resolve_model_path(name: str) -> Path | None:
    spec = get_spec(name)
    if not _is_downloaded(spec):
        return None
    if spec.backend == "whisper":
        return _whisper_checkpoint(name)
    return _target_dir(name)


def is_model_downloaded(name: str) -> bool:
    return _is_downloaded(get_spec(name))


def is_diarization_model_downloaded() -> bool:
    return is_model_downloaded(DIARIZATION_MODEL)


def diarization_model_path() -> Path | None:
    return resolve_model_path(DIARIZATION_MODEL)


def model_size_mb(name: str) -> int:
    return get_spec(name).size_mb


def model_kind(name: str) -> AsrModelKind:
    return get_spec(name).kind


def model_backend(name: str) -> _Backend:
    return get_spec(name).backend


def download_model(name: str) -> Path:
    spec = get_spec(name)
    if spec.backend == "whisper":
        return _download_whisper(name)
    assert spec.modelscope_id is not None
    return _download_modelscope(spec.modelscope_id, _target_dir(name))


__all__ = [
    "DIARIZATION_MODEL",
    "PUNCTUATION_MODEL",
    "ModelSpec",
    "VAD_MODEL_ID",
    "download_model",
    "get_spec",
    "is_diarization_model_downloaded",
    "is_model_downloaded",
    "list_specs",
    "model_backend",
    "model_kind",
    "model_size_mb",
    "resolve_model_path",
]
