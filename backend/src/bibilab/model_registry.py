"""Unified model dependency registry.

All non-LLM model downloads flow through ensure() with per-model locks
and atomic .partial → rename to prevent concurrent-download corruption.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bibilab.config import BibilabConfig, models_dir

logger = logging.getLogger(__name__)

ModelKind = Literal["transcription", "vad", "diarization", "embedding", "reranker", "punctuation"]
Backend = Literal["http_files", "http_archive"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    kind: ModelKind
    backend: Backend
    size_mb: int
    integrity_files: list[str]  # rel paths within target dir that must exist post-download
    local_subdir: str  # relative to models_dir()
    http_files: list[tuple[str, str]] | None = None  # [(url, rel_path), ...]

    def __post_init__(self) -> None:
        if not self.integrity_files:
            raise ValueError(f"{self.id!r}: integrity_files must be non-empty")


# ---- Spec definitions ------------------------------------------------

_SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

_SPECS: dict[str, ModelSpec] = {
    "sherpa-sensevoice": ModelSpec(
        id="sherpa-sensevoice",
        display_name="sherpa-onnx SenseVoice",
        kind="transcription",
        backend="http_archive",
        size_mb=1048,
        integrity_files=["model.int8.onnx", "tokens.txt"],
        local_subdir="asr/sherpa-sensevoice",
        http_files=[
            (
                f"{_SHERPA_RELEASE}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            )
        ],
    ),
    "sherpa-whisper-large-v3": ModelSpec(
        id="sherpa-whisper-large-v3",
        display_name="sherpa-onnx Whisper large-v3 (int8)",
        kind="transcription",
        backend="http_archive",
        size_mb=1068,
        integrity_files=["large-v3-encoder.int8.onnx", "large-v3-decoder.int8.onnx", "large-v3-tokens.txt"],
        local_subdir="asr/sherpa-whisper-large-v3",
        http_files=[
            (
                f"{_SHERPA_RELEASE}/asr-models/sherpa-onnx-whisper-large-v3.tar.bz2",
                "sherpa-onnx-whisper-large-v3.tar.bz2",
            )
        ],
    ),
    "sherpa-ct-punc": ModelSpec(
        id="sherpa-ct-punc",
        display_name="sherpa-onnx CT-Transformer Punctuation (zh-en)",
        kind="punctuation",
        backend="http_archive",
        size_mb=279,
        integrity_files=["model.onnx", "tokens.json"],
        local_subdir="asr/sherpa-ct-punc",
        http_files=[
            (
                f"{_SHERPA_RELEASE}/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
                "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
            )
        ],
    ),
    "sherpa-silero-vad": ModelSpec(
        id="sherpa-silero-vad",
        display_name="Silero VAD",
        kind="vad",
        backend="http_files",
        size_mb=1,
        integrity_files=["silero_vad.onnx"],
        local_subdir="asr/sherpa-silero-vad",
        http_files=[(f"{_SHERPA_RELEASE}/asr-models/silero_vad.onnx", "silero_vad.onnx")],
    ),
    "sherpa-campplus": ModelSpec(
        id="sherpa-campplus",
        display_name="CAM++ (sherpa-onnx, Speaker Diarization)",
        kind="diarization",
        backend="http_files",
        size_mb=28,
        integrity_files=["3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"],
        local_subdir="asr/sherpa-campplus",
        http_files=[
            (
                f"{_SHERPA_RELEASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
            )
        ],
    ),
    "multilingual-e5-small": ModelSpec(
        id="multilingual-e5-small",
        display_name="Multilingual Embedding (e5-small)",
        kind="embedding",
        backend="http_files",
        size_mb=448,
        integrity_files=["onnx/model.onnx", "onnx/tokenizer.json"],
        local_subdir="embedding/intfloat_multilingual-e5-small",
        http_files=[
            (
                "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx",
                "onnx/model.onnx",
            ),
            (
                "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/tokenizer.json",
                "onnx/tokenizer.json",
            ),
        ],
    ),
    # Sole reranker: int8 quantized bge-reranker-base — ~4× smaller (266 MiB) and
    # ~1.8× faster on CPU than fp32, with the quality delta (top-8 91% vs fp32)
    # absorbed by the gateless top-k arch ("rerank is ordering, not authority").
    # fp32 was dropped: with the session pinned to a kernel EP (no CoreML), fp32
    # bought only marginal quality at 4× size + ~1.85× latency. The remote file is
    # model_quantized.onnx, stored as model.onnx so the loader stays filename-agnostic.
    "bge-reranker-base-q": ModelSpec(
        id="bge-reranker-base-q",
        display_name="bge-reranker-base int8 (Cross-encoder)",
        kind="reranker",
        backend="http_files",
        size_mb=266,
        integrity_files=["model.onnx", "tokenizer.json"],
        local_subdir="reranker/Xenova_bge-reranker-base-q",
        http_files=[
            (
                "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx/model_quantized.onnx",
                "model.onnx",
            ),
            (
                "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/tokenizer.json",
                "tokenizer.json",
            ),
        ],
    ),
}

EMBEDDING_SPEC_ID = "multilingual-e5-small"
RERANKER_SPEC_ID = "bge-reranker-base-q"

# sherpa-onnx spec ids — the ones transcribe.py and punctuate.py actually run on.
SHERPA_SENSEVOICE_SPEC_ID = "sherpa-sensevoice"
SHERPA_WHISPER_SPEC_ID = "sherpa-whisper-large-v3"
SHERPA_PUNC_SPEC_ID = "sherpa-ct-punc"
SHERPA_VAD_SPEC_ID = "sherpa-silero-vad"
SHERPA_DIARIZATION_SPEC_ID = "sherpa-campplus"

# TranscriptionConfig.model values ("large-v3", "sensevoice-small") are stable public
# config strings; which concrete spec backs them is an implementation detail resolved
# once here, shared by required_models() below and transcribe.py's engine loader.
_TRANSCRIPTION_SPEC_BY_MODEL = {
    "large-v3": SHERPA_WHISPER_SPEC_ID,
    "sensevoice-small": SHERPA_SENSEVOICE_SPEC_ID,
}


def resolve_transcription_spec_id(model: str) -> str:
    return _TRANSCRIPTION_SPEC_BY_MODEL.get(model, model)


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


def _download_http_archive(spec: ModelSpec, target: Path) -> None:
    """Stream a .tar.bz2, extract it, and flatten its single top-level directory
    into target. Every k2-fsa release archive ships exactly one top-level dir;
    a violation fails loud rather than silently nesting the wrong layout."""
    assert spec.http_files is not None and len(spec.http_files) == 1
    url, archive_name = spec.http_files[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    import httpx  # noqa: PLC0415

    try:
        archive_path = tmp / archive_name
        logger.info("Downloading %s → %s", url, archive_path)
        # Archives run into the hundreds of MB; httpx's 5s default (connect/read/write/
        # pool) is tuned for API calls, not large streamed downloads.
        timeout = httpx.Timeout(10.0, read=60.0)
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f.write(chunk)

        with tarfile.open(archive_path) as tf:
            tf.extractall(tmp, filter="data")
        archive_path.unlink()

        entries = list(tmp.iterdir())
        assert len(entries) == 1 and entries[0].is_dir(), (
            f"{spec.id}: expected exactly one top-level directory in the archive, got {entries}"
        )
        extracted = entries[0]
        for item in extracted.iterdir():
            item.rename(tmp / item.name)
        extracted.rmdir()
    except Exception:
        logger.exception("Archive download/extraction failed for %s", spec.id)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    try:
        tmp.rename(target)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"atomic rename failed for {spec.id}: {exc}") from exc
    logger.info("Model extracted to %s", target)


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

        if spec.backend == "http_files":
            _download_http_files(spec, target)
        elif spec.backend == "http_archive":
            _download_http_archive(spec, target)
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
            specs.append(get_spec(resolve_transcription_spec_id(model)))
        except ValueError:
            logger.warning("Unknown transcription model %r — skipping in required-models check", model)
    specs.append(get_spec(SHERPA_VAD_SPEC_ID))
    specs.append(get_spec(SHERPA_DIARIZATION_SPEC_ID))
    specs.append(get_spec(SHERPA_PUNC_SPEC_ID))
    specs.append(get_spec(EMBEDDING_SPEC_ID))
    if cfg.rag.reranking_enabled:
        specs.append(get_spec(RERANKER_SPEC_ID))
    return specs


def missing_required_models(cfg: BibilabConfig) -> list[str]:
    """Return spec IDs that are required but not present on disk."""
    return [s.id for s in required_models(cfg) if not _integrity_ok(s)]


__all__ = [
    "EMBEDDING_SPEC_ID",
    "ModelKind",
    "ModelSpec",
    "RERANKER_SPEC_ID",
    "SHERPA_DIARIZATION_SPEC_ID",
    "SHERPA_PUNC_SPEC_ID",
    "SHERPA_SENSEVOICE_SPEC_ID",
    "SHERPA_VAD_SPEC_ID",
    "SHERPA_WHISPER_SPEC_ID",
    "ensure",
    "get_spec",
    "list_specs",
    "missing_required_models",
    "resolve_transcription_spec_id",
    "required_models",
]
