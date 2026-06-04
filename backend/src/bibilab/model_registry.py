"""Unified model dependency registry.

All non-LLM model downloads flow through ensure() with per-model locks
and atomic .partial → rename to prevent concurrent-download corruption.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bibilab.config import BibilabConfig, models_dir

logger = logging.getLogger(__name__)

ModelKind = Literal["transcription", "vad", "diarization", "embedding", "reranker", "punctuation"]
Backend = Literal["http_files", "modelscope", "whisper_warp"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    kind: ModelKind
    backend: Backend
    size_mb: int
    integrity_files: list[str]  # rel paths within target dir that must exist post-download
    local_subdir: str  # relative to models_dir()
    modelscope_id: str | None = None
    http_files: list[tuple[str, str]] | None = None  # [(url, rel_path), ...]

    def __post_init__(self) -> None:
        if not self.integrity_files:
            raise ValueError(f"{self.id!r}: integrity_files must be non-empty")


# ---- Spec definitions ------------------------------------------------

_SPECS: dict[str, ModelSpec] = {
    "large-v3": ModelSpec(
        id="large-v3",
        display_name="Faster Whisper large-v3",
        kind="transcription",
        backend="whisper_warp",
        size_mb=3000,
        integrity_files=["large-v3.pt"],
        local_subdir="asr/whisper",
    ),
    "sensevoice-small": ModelSpec(
        id="sensevoice-small",
        display_name="SenseVoice Small",
        kind="transcription",
        backend="modelscope",
        size_mb=936,
        integrity_files=["configuration.json"],
        local_subdir="asr/sensevoice-small",
        modelscope_id="iic/SenseVoiceSmall",
    ),
    "cam++": ModelSpec(
        id="cam++",
        display_name="CAM++ (Speaker Diarization)",
        kind="diarization",
        backend="modelscope",
        size_mb=28,
        integrity_files=["configuration.json"],
        local_subdir="asr/cam++",
        modelscope_id="iic/speech_campplus_sv_zh-cn_16k-common",
    ),
    "fsmn-vad": ModelSpec(
        id="fsmn-vad",
        display_name="FSMN-VAD (Voice Activity Detection)",
        kind="vad",
        backend="modelscope",
        size_mb=4,
        integrity_files=["configuration.json"],
        local_subdir="asr/fsmn-vad",
        modelscope_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
    "ct-punc": ModelSpec(
        id="ct-punc",
        display_name="CT-Transformer Punctuation (zh-en)",
        kind="punctuation",
        backend="modelscope",
        size_mb=1050,
        integrity_files=["configuration.json"],
        local_subdir="asr/ct-punc",
        modelscope_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    ),
    "multilingual-e5": ModelSpec(
        id="multilingual-e5",
        display_name="Multilingual Embedding (MiniLM-L12-v2)",
        kind="embedding",
        backend="http_files",
        size_mb=420,
        integrity_files=["onnx/model.onnx", "onnx/tokenizer.json"],
        local_subdir="embedding",
        http_files=[
            (
                "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx",
                "onnx/model.onnx",
            ),
            (
                "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/tokenizer.json",
                "onnx/tokenizer.json",
            ),
        ],
    ),
    "bge-reranker-base": ModelSpec(
        id="bge-reranker-base",
        display_name="bge-reranker-base (Cross-encoder)",
        kind="reranker",
        backend="http_files",
        size_mb=280,
        integrity_files=["model.onnx", "tokenizer.json"],
        local_subdir="reranker/Xenova_bge-reranker-base",
        http_files=[
            (
                "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model.onnx",
                "model.onnx",
            ),
            (
                "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json",
                "tokenizer.json",
            ),
        ],
    ),
}

EMBEDDING_SPEC_ID = "multilingual-e5"
RERANKER_SPEC_ID = "bge-reranker-base"
DIARIZATION_SPEC_ID = "cam++"
VAD_SPEC_ID = "fsmn-vad"
PUNC_SPEC_ID = "ct-punc"
WHISPER_SPEC_ID = "large-v3"


def list_specs() -> list[ModelSpec]:
    return list(_SPECS.values())


def get_spec(spec_id: str) -> ModelSpec:
    if spec_id not in _SPECS:
        raise ValueError(f"Unknown model {spec_id!r}")
    return _SPECS[spec_id]


# ---- Path resolution -------------------------------------------------


def _target_dir(spec: ModelSpec) -> Path:
    return models_dir(spec.local_subdir)


def _integrity_ok(spec: ModelSpec) -> bool:
    target = _target_dir(spec)
    for f in spec.integrity_files:
        if not (target / f).exists():
            return False
    return True


# ---- Download backends -----------------------------------------------


def _download_modelscope(spec: ModelSpec, target: Path) -> None:
    assert spec.modelscope_id is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    from modelscope.hub.snapshot_download import snapshot_download  # noqa: PLC0415

    logger.info("Downloading model from ModelScope: %s", spec.modelscope_id)
    try:
        snapshot_download(spec.modelscope_id, local_dir=str(tmp))
    except Exception:
        logger.exception("ModelScope download failed for %s", spec.modelscope_id)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    try:
        tmp.rename(target)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"atomic rename failed for {spec.id}: {exc}") from exc
    logger.info("Model downloaded to %s", target)


def _download_http_files(spec: ModelSpec, target: Path) -> None:
    assert spec.http_files is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    import httpx  # noqa: PLC0415

    try:
        for url, rel_path in spec.http_files:
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s → %s", url, dest)
            with httpx.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(1024 * 1024):
                        f.write(chunk)
    except Exception:
        logger.exception("HTTP download failed for %s", spec.id)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    try:
        tmp.rename(target)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"atomic rename failed for {spec.id}: {exc}") from exc
    logger.info("Model downloaded to %s", target)


def _download_whisper_warp(spec: ModelSpec, target: Path) -> None:
    # funasr 1.3.7's openai branch hardcodes whisper.load_model(name) with no
    # download_root, so it always writes to ~/.cache/whisper. Bypass it and call
    # openai-whisper's documented public API directly. See issue #426.
    import whisper  # noqa: PLC0415  # openai-whisper (lazy: pulls in torch)

    logger.info("Downloading Whisper large-v3 (~3 GB) via WhisperWarp — this may take several minutes")
    try:
        whisper.load_model(spec.id, download_root=str(target))
    except Exception:
        logger.exception("Whisper large-v3 download failed")
        raise
    logger.info("Whisper large-v3 download complete → %s", target)


# ---- Unified download entry point ------------------------------------


_inflight: dict[str, threading.Lock] = {}


def ensure(spec_id: str) -> Path:
    """Return target dir for *spec_id*, downloading first if needed.

    Per-model threading.Lock + atomic .partial → rename prevents
    concurrent-download corruption within a single process.
    """
    spec = get_spec(spec_id)
    target = _target_dir(spec)

    if _integrity_ok(spec):
        return target

    lock = _inflight.setdefault(spec_id, threading.Lock())
    with lock:
        if _integrity_ok(spec):
            return target

        if spec.backend == "modelscope":
            _download_modelscope(spec, target)
        elif spec.backend == "http_files":
            _download_http_files(spec, target)
        elif spec.backend == "whisper_warp":
            _download_whisper_warp(spec, target)
        else:
            raise ValueError(f"Unknown backend {spec.backend!r} for {spec_id!r}")

        if not _integrity_ok(spec):
            raise RuntimeError(f"download completed but integrity check failed for {spec_id!r}")

    return target


# ---- Config-driven helpers -------------------------------------------


def required_models(cfg: BibilabConfig) -> list[ModelSpec]:
    """Return model specs required under the current config."""
    specs: list[ModelSpec] = []
    model = cfg.transcription.model
    if model is not None:
        try:
            specs.append(get_spec(model))
        except ValueError:
            logger.warning("Unknown transcription model %r — skipping in required-models check", model)
    specs.append(get_spec(VAD_SPEC_ID))
    specs.append(get_spec(DIARIZATION_SPEC_ID))
    specs.append(get_spec(PUNC_SPEC_ID))
    specs.append(get_spec(EMBEDDING_SPEC_ID))
    if cfg.rag.reranking_enabled:
        specs.append(get_spec(RERANKER_SPEC_ID))
    return specs


def missing_required_models(cfg: BibilabConfig) -> list[str]:
    """Return spec IDs that are required but not present on disk."""
    return [s.id for s in required_models(cfg) if not _integrity_ok(s)]


__all__ = [
    "DIARIZATION_SPEC_ID",
    "EMBEDDING_SPEC_ID",
    "ModelKind",
    "ModelSpec",
    "PUNC_SPEC_ID",
    "RERANKER_SPEC_ID",
    "VAD_SPEC_ID",
    "WHISPER_SPEC_ID",
    "ensure",
    "get_spec",
    "list_specs",
    "missing_required_models",
    "required_models",
]
