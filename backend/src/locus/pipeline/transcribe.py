"""Faster Whisper transcription step."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from locus.config import TranscriptionConfig, locus_home

logger = logging.getLogger(__name__)

# Module-level singleton — avoid reloading the model on every job
_model = None
_model_key: tuple[str, str] | None = None  # (model_size, device)


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


def _load_model(cfg: TranscriptionConfig):
    global _model, _model_key

    from faster_whisper import WhisperModel  # noqa: PLC0415

    key = (cfg.model_size, cfg.device)
    if _model is None or _model_key != key:
        logger.info("Loading Whisper model %s on %s", cfg.model_size, cfg.device)
        _model = WhisperModel(cfg.model_size, device=cfg.device, compute_type="float16")
        _model_key = key
    return _model


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> list[WhisperSegment]:
    model = _load_model(cfg)
    language = None if cfg.language == "auto" else cfg.language
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        language=language,
    )
    return [WhisperSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]


def write_transcript(segments: list[WhisperSegment], video_id: str) -> Path:
    """Write segments to ~/.locus/transcripts/{video_id}.txt, one line per segment."""
    transcripts_dir = locus_home() / "transcripts"
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
