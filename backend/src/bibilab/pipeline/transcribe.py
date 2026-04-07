"""Faster Whisper transcription step."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from bibilab.config import TranscriptionConfig, bibilab_home
from bibilab.whisper_models import download_whisper_model, resolve_local_model_path

logger = logging.getLogger(__name__)

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


def _load_model(cfg: TranscriptionConfig):
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
