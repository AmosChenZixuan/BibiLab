"""Transcription stage.

Dispatches by model name. `large-v3` uses openai-whisper; everything else flows
through FunASR. Speaker diarization (CAM++) is mandatory:

- SenseVoice runs through FunASR with `spk_model="cam++"`, so each
  `sentence_info` entry carries an `spk` label inline.
- Whisper has no native FunASR spk path, so a separate FunASR
  vad+spk-only pass produces speaker spans that get overlap-merged
  onto the Whisper segments.

CAM++ auto-downloads on first ingest, same pattern as the embedding/reranker
models — no settings page button required.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibilab.asr_models import (
    DIARIZATION_MODEL,
    PUNCTUATION_MODEL,
    VAD_MODEL_ID,
    download_model,
    model_backend,
    resolve_model_path,
)
from bibilab.config import TranscriptionConfig, bibilab_home
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# SenseVoice emits per-segment tags: <|zh|>, <|NEUTRAL|>, <|BGM|>, <|withitn|>, etc.
_FUNASR_TAG_RE = re.compile(r"<\|[^|]*\|>\s*")


def _strip_funasr_tags(text: str) -> str:
    return _FUNASR_TAG_RE.sub("", text)


_whisper_model: Any = None
_whisper_key: tuple[str, str] | None = None  # (model, device)

_funasr_pipeline: Any = None
_funasr_key: tuple[str, str] | None = None  # (model, device); always spk-enabled

_diarize_pipeline: Any = None
_diarize_device: str | None = None


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class _SpeakerSpan:
    start: float
    end: float
    speaker: str


def _ensure_downloaded(name: str) -> Path:
    path = resolve_model_path(name)
    if path is None:
        download_model(name)
        path = resolve_model_path(name)
        if path is None:
            raise PipelineError(f"Model {name!r} missing after download")
    return path


# ---------------------------------------------------------------------------
# Whisper (openai-whisper)
# ---------------------------------------------------------------------------


def _load_whisper(cfg: TranscriptionConfig) -> Any:
    global _whisper_model, _whisper_key
    import whisper  # noqa: PLC0415

    key = (cfg.model, cfg.device)
    if _whisper_model is not None and _whisper_key == key:
        return _whisper_model

    checkpoint = _ensure_downloaded(cfg.model)
    logger.info("Loading Whisper model %s on %s", cfg.model, cfg.device)
    _whisper_model = whisper.load_model(str(checkpoint), device=cfg.device)
    _whisper_key = key
    return _whisper_model


def _transcribe_whisper(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    model = _load_whisper(cfg)
    language = None if cfg.language == "auto" else cfg.language
    result = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=cfg.beam_size,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    segments: list[WhisperSegment] = []
    for s in result.get("segments", []):
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        segments.append(WhisperSegment(start=float(s["start"]), end=float(s["end"]), text=text))
    detected = result.get("language") if cfg.language == "auto" else cfg.language
    return segments, detected


# ---------------------------------------------------------------------------
# Diarization (FunASR vad + CAM++ only, used for Whisper path)
# ---------------------------------------------------------------------------


def _load_diarize(device: str) -> Any:
    global _diarize_pipeline, _diarize_device
    from funasr import AutoModel  # noqa: PLC0415

    if _diarize_pipeline is not None and _diarize_device == device:
        return _diarize_pipeline

    actual_device = "cuda:0" if device == "cuda" else "cpu"
    spk_path = _ensure_downloaded(DIARIZATION_MODEL)
    logger.info("Loading diarization pipeline (CAM++) on %s", actual_device)
    _diarize_pipeline = AutoModel(
        model=None,
        vad_model=VAD_MODEL_ID,
        spk_model=str(spk_path),
        spk_mode="vad_segment",
        device=actual_device,
        disable_update=True,
        disable_pbar=True,
    )
    _diarize_device = device
    return _diarize_pipeline


def _diarize(audio_path: Path, device: str) -> list[_SpeakerSpan]:
    model = _load_diarize(device)
    res = model.generate(input=str(audio_path))
    if not res:
        return []
    raw = res[0].get("sentence_info") or res[0].get("segments") or []
    spans: list[_SpeakerSpan] = []
    for s in raw:
        start = _coerce_seconds(float(s.get("start", 0.0)))
        end = _coerce_seconds(float(s.get("end", 0.0)))
        spk = s.get("spk", s.get("speaker"))
        if spk is None:
            continue
        spans.append(_SpeakerSpan(start=start, end=end, speaker=f"SPK_{spk}"))
    spans.sort(key=lambda s: s.start)
    return spans


def _best_speaker(seg: WhisperSegment, spans: list[_SpeakerSpan]) -> str | None:
    best: str | None = None
    best_overlap = 0.0
    for sp in spans:
        overlap = max(0.0, min(seg.end, sp.end) - max(seg.start, sp.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = sp.speaker
    return best if best_overlap > 0 else None


# ---------------------------------------------------------------------------
# FunASR (SenseVoice + inline CAM++ via spk_model)
# ---------------------------------------------------------------------------


def _load_funasr(cfg: TranscriptionConfig) -> Any:
    global _funasr_pipeline, _funasr_key
    from funasr import AutoModel  # noqa: PLC0415

    key = (cfg.model, cfg.device)
    if _funasr_pipeline is not None and _funasr_key == key:
        return _funasr_pipeline

    device = "cuda:0" if cfg.device == "cuda" else "cpu"
    model_path = _ensure_downloaded(cfg.model)
    spk_path = _ensure_downloaded(DIARIZATION_MODEL)
    punc_path = _ensure_downloaded(PUNCTUATION_MODEL)
    logger.info("Loading FunASR model %s on %s (+CAM++)", cfg.model, device)
    _funasr_pipeline = AutoModel(
        model=str(model_path),
        device=device,
        vad_model=VAD_MODEL_ID,
        punc_model=str(punc_path),
        spk_model=str(spk_path),
        spk_mode="vad_segment",
        disable_update=True,
        disable_pbar=True,
    )
    _funasr_key = key
    return _funasr_pipeline


def _coerce_seconds(value: float) -> float:
    # FunASR sentence_info entries report seconds for SenseVoice but milliseconds
    # in some spk-enabled paths. Anything past an hour is almost certainly ms.
    return value / 1000.0 if value > 3600 else value


def _transcribe_funasr(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    model = _load_funasr(cfg)
    gen_kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "use_itn": True,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    if cfg.language and cfg.language != "auto":
        gen_kwargs["language"] = cfg.language

    res = model.generate(**gen_kwargs)
    if not res:
        logger.warning("FunASR returned no results for %s", audio_path)
        return [], None

    first = res[0]
    raw = first.get("sentence_info") or first.get("segments") or []
    segments: list[WhisperSegment] = []
    for s in raw:
        text = _strip_funasr_tags(str(s.get("text") or s.get("sentence") or "")).strip()
        if not text:
            continue
        start = _coerce_seconds(float(s.get("start", 0.0)))
        end = _coerce_seconds(float(s.get("end", 0.0)))
        spk = s.get("spk")
        speaker = f"SPK_{spk}" if spk is not None else None
        segments.append(WhisperSegment(start=start, end=end, text=text, speaker=speaker))

    detected = first.get("language")
    if cfg.language and cfg.language != "auto":
        detected = cfg.language
    elif detected == "auto":
        detected = None
    return segments, detected


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    """Transcribe audio. Returns (segments, detected_language). Segments carry
    speaker labels — inline for FunASR, overlap-merged for Whisper."""
    try:
        backend = model_backend(cfg.model)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    if backend == "whisper":
        segments, detected = _transcribe_whisper(audio_path, cfg)
        spans = _diarize(audio_path, cfg.device)
        if spans:
            for seg in segments:
                seg.speaker = _best_speaker(seg, spans)
        return segments, detected
    return _transcribe_funasr(audio_path, cfg)


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


__all__ = [
    "WhisperSegment",
    "transcribe",
    "write_transcript",
]
