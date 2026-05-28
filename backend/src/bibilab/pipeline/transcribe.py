"""Transcription step — Whisper + SenseVoice dispatch + speaker diarization."""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import BatchedInferencePipeline

from bibilab.asr_models import (
    is_diarization_model_downloaded,
    resolve_model_path,
)
from bibilab.config import TranscriptionConfig, bibilab_home

logger = logging.getLogger(__name__)


def _preload_bundled_cuda_libs() -> None:
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

# Module-level singletons
_whisper_pipeline = None
_whisper_key: tuple[str, str] | None = None  # (model_size, device)


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def _compute_type_for_device(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _load_whisper(cfg: TranscriptionConfig) -> "BatchedInferencePipeline":
    global _whisper_pipeline, _whisper_key

    from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: PLC0415

    key = (cfg.model_size, cfg.device)
    if _whisper_pipeline is None or _whisper_key != key:
        local_path = resolve_model_path("whisper", cfg.model_size)
        model_source = str(local_path) if local_path is not None else cfg.model_size
        if local_path is None:
            from bibilab.asr_models import download_model

            download_model("whisper", cfg.model_size)
        compute_type = _compute_type_for_device(cfg.device)
        logger.info(
            "Loading Whisper model %s on %s (%s)",
            cfg.model_size,
            cfg.device,
            compute_type,
        )
        model = WhisperModel(
            model_source,
            device=cfg.device,
            compute_type=compute_type,
            cpu_threads=4,
        )
        logger.info(
            "Whisper encoder runs on %s (%s), decoder on CPU (%d threads)",
            cfg.device,
            compute_type,
            4,
        )
        _whisper_pipeline = BatchedInferencePipeline(model)
        _whisper_key = key
    return _whisper_pipeline


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

_SILENCE_PROB_THRESHOLD = 0.6


def _transcribe_whisper(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    pipeline = _load_whisper(cfg)
    language = None if cfg.language == "auto" else cfg.language
    batch_size = 16 if cfg.device == "cuda" else 1
    segments, info = pipeline.transcribe(
        str(audio_path),
        beam_size=cfg.beam_size,
        batch_size=batch_size,
        vad_filter=True,
        language=language,
    )
    segment_list = [
        WhisperSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
        if not (s.no_speech_prob > _SILENCE_PROB_THRESHOLD)
    ]
    detected_language: str | None = None if cfg.language != "auto" else info.language
    return segment_list, detected_language


# ---------------------------------------------------------------------------
# Speaker merge
# ---------------------------------------------------------------------------


def _best_speaker(seg: WhisperSegment, speaker_segs: list) -> str | None:
    """Return speaker with maximum timestamp overlap with seg, or None."""
    best = None
    best_overlap = 0.0
    for spk in speaker_segs:
        overlap = max(0.0, min(seg.end, spk.end) - max(seg.start, spk.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = spk.speaker
    return best if best_overlap > 0 else None


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    """Transcribe audio to segments. Returns (segments, detected_language)."""

    # Diarization pre-pass (engine-agnostic, if model downloaded)
    speaker_segs: list = []
    if is_diarization_model_downloaded():
        from bibilab.pipeline.diarize import diarize

        speaker_segs = diarize(audio_path, cfg.device)

    # Engine dispatch
    if cfg.engine == "sensevoice":
        from bibilab.pipeline.sensevoice import _transcribe_sensevoice

        segments, detected_language = _transcribe_sensevoice(audio_path, cfg)
    else:
        segments, detected_language = _transcribe_whisper(audio_path, cfg)

    # Merge speaker labels
    if speaker_segs:
        for seg in segments:
            seg.speaker = _best_speaker(seg, speaker_segs)

    return segments, detected_language


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


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
        line = f"[{h:02d}:{m:02d}:{s:02d}] {seg.text}"
        if seg.speaker:
            line += f" [{seg.speaker}]"
        lines.append(line)

    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, out_path)
    logger.info("Wrote %d segments to %s", len(segments), out_path)
    return out_path
