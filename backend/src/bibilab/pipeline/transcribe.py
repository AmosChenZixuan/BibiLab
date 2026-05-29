"""Transcription stage.

All ASR models flow through FunASR's AutoModel — SenseVoice natively, Whisper
via FunASR's WhisperWarp wrapper. Speaker diarization (CAM++) is mandatory for
both paths. Punctuation is native to the ASR: SenseVoice emits `，。？！` from
its training transcripts; Whisper-large-v3 emits English punctuation.

When punc_model is absent, FunASR auto-falls-back to vad_segment for speaker
segmentation (auto_model.py:810-812), so each `sentence_info` entry carries
an `spk` label inline.

VAD is tuned (`max_single_segment_time=15000`, `max_end_silence_time=500`)
to keep VAD chunks ≤15s — bounds segment length, gives finer speaker
granularity, and prevents the 60s hard-cap that used to slice through words
on continuous BGM-laden audio.

CAM++ auto-downloads on first ingest, same pattern as the embedding/reranker
models.
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
    VAD_MODEL,
    download_model,
    get_spec,
    resolve_model_path,
)
from bibilab.config import TranscriptionConfig, bibilab_home
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# SenseVoice emits per-segment tags: <|zh|>, <|NEUTRAL|>, <|BGM|>, <|withitn|>, etc.
_FUNASR_TAG_RE = re.compile(r"<\|[^|]*\|>\s*")


def _strip_funasr_tags(text: str) -> str:
    return _FUNASR_TAG_RE.sub("", text)


_funasr_pipeline: Any = None
_funasr_key: tuple[str, str] | None = None  # (model, device)


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def _ensure_downloaded(name: str) -> Path:
    path = resolve_model_path(name)
    if path is None:
        download_model(name)
        path = resolve_model_path(name)
        if path is None:
            raise PipelineError(f"Model {name!r} missing after download")
    return path


# ---------------------------------------------------------------------------
# FunASR (all models via AutoModel; SenseVoice native, Whisper via WhisperWarp)
# ---------------------------------------------------------------------------


def _load_funasr(cfg: TranscriptionConfig) -> Any:
    global _funasr_pipeline, _funasr_key
    from funasr import AutoModel  # noqa: PLC0415

    key = (cfg.model, cfg.device)
    if _funasr_pipeline is not None and _funasr_key == key:
        return _funasr_pipeline

    device = "cuda:0" if cfg.device == "cuda" else "cpu"
    spk_path = _ensure_downloaded(DIARIZATION_MODEL)
    vad_path = _ensure_downloaded(VAD_MODEL)

    if cfg.model == "large-v3":
        # WhisperWarp wraps openai-whisper inside AutoModel.
        # hub="openai" tells WhisperWarp to use openai-whisper internally;
        # the model auto-downloads to ~/.cache/whisper/ on first use.
        automodel_kwargs = {"model": "Whisper-large-v3", "hub": "openai"}
    else:
        model_path = _ensure_downloaded(cfg.model)
        automodel_kwargs = {"model": str(model_path)}

    logger.info("Loading FunASR model %s on %s (+CAM++)", cfg.model, device)
    _funasr_pipeline = AutoModel(
        device=device,
        vad_model=str(vad_path),
        vad_kwargs={"max_single_segment_time": 15000, "max_end_silence_time": 500},
        spk_model=str(spk_path),
        disable_update=True,
        disable_pbar=True,
        **automodel_kwargs,
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
    """Transcribe audio. Returns (segments, detected_language).

    Segments carry speaker labels inline from FunASR's CAM++ diarization.
    """
    try:
        get_spec(cfg.model)  # raises ValueError on unknown model
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
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
