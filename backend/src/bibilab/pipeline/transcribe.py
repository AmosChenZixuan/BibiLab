"""Faster Whisper transcription step."""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

from bibilab.config import TranscriptionConfig, bibilab_home
from bibilab.whisper_models import download_whisper_model, resolve_local_model_path

logger = logging.getLogger(__name__)


def _preload_bundled_cuda_libs() -> None:
    # ctranslate2 loads CUDA libs via dlopen(soname). Soname resolution uses
    # the search-path list cached by ld.so at process startup — mutating
    # LD_LIBRARY_PATH post-startup does not affect it. Preload bundled libs
    # by absolute path with RTLD_GLOBAL so their symbols are visible when
    # ctranslate2 later calls dlopen.
    for pkg, soname in (
        ("nvidia.cublas", "libcublas.so.12"),
        ("nvidia.cudnn", "libcudnn.so.9"),
    ):
        try:
            mod = __import__(pkg, fromlist=[""])
            lib = Path(mod.__path__[0]) / "lib" / soname
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                logger.debug("preloaded %s", lib)
        except (ImportError, OSError) as exc:
            logger.debug("skip preload %s: %s", soname, exc)


_preload_bundled_cuda_libs()

# Module-level singleton — avoid reloading the model on every job
_model = None
_model_key: tuple[str, str] | None = None  # (model_size, device)


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


def _compute_type_for_device(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _load_model(cfg: TranscriptionConfig) -> "WhisperModel":
    global _model, _model_key

    from faster_whisper import WhisperModel  # noqa: PLC0415

    key = (cfg.model_size, cfg.device)
    if _model is None or _model_key != key:
        local_path = resolve_local_model_path(cfg.model_size)
        model_source = str(local_path) if local_path is not None else cfg.model_size
        if local_path is None:
            download_whisper_model(cfg.model_size)
        logger.info(
            "Loading Whisper model %s on %s from %s",
            cfg.model_size,
            cfg.device,
            model_source,
        )
        _model = WhisperModel(
            model_source,
            device=cfg.device,
            compute_type=_compute_type_for_device(cfg.device),
        )
        _model_key = key
    return _model


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    """Transcribe audio to segments. Returns (segments, detected_language)."""
    model = _load_model(cfg)
    language = None if cfg.language == "auto" else cfg.language
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=cfg.beam_size,
        vad_filter=True,
        language=language,
    )
    segment_list = [WhisperSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
    detected_language: str | None = None if cfg.language != "auto" else info.language
    return segment_list, detected_language


def write_transcript(segments: list[WhisperSegment], video_id: str) -> Path:
    """Write segments to ~/.bibilab/transcripts/{video_id}.txt, one line per segment."""
    transcripts_dir = bibilab_home() / "transcripts"
    out_path = transcripts_dir / f"{video_id}.txt"
    tmp = out_path.with_suffix(".tmp")

    lines = []
    for seg in segments:
        h = int(seg.start) // 3600
        m = (int(seg.start) % 3600) // 60
        s = int(seg.start) % 60
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {seg.text}")

    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, out_path)
    logger.info("Wrote %d segments to %s", len(segments), out_path)
    return out_path
