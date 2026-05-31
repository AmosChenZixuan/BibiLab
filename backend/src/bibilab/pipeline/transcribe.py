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
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibilab.config import TranscriptionConfig
from bibilab.model_registry import DIARIZATION_SPEC_ID, VAD_SPEC_ID, ensure, get_spec
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# SenseVoice emits per-segment tags: <|zh|>, <|NEUTRAL|>, <|BGM|>, <|withitn|>, etc.
_FUNASR_TAG_RE = re.compile(r"<\|[^|]*\|>\s*")


def _strip_funasr_tags(text: str) -> str:
    return _FUNASR_TAG_RE.sub("", text)


_funasr_pipeline: Any = None
_funasr_key: tuple[str, str] | None = None  # (model, device)

# Serializes the shared FunASR instance (load + generate). generate() mutates
# internal state, so concurrent calls (worker max_concurrent_jobs > 1) race and
# corrupt output. Transcription is also CPU/GIL-bound with the GPU only ~17%
# busy, so parallel transcribes give no throughput gain anyway — serializing
# costs nothing and is safe.
_transcribe_lock = threading.Lock()


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def _load_funasr(cfg: TranscriptionConfig) -> Any:
    global _funasr_pipeline, _funasr_key
    from funasr import AutoModel  # noqa: PLC0415

    key = (cfg.model, cfg.device)
    if _funasr_pipeline is not None and _funasr_key == key:
        return _funasr_pipeline

    device = "cuda:0" if cfg.device == "cuda" else "cpu"
    spk_path = ensure(DIARIZATION_SPEC_ID)
    vad_path = ensure(VAD_SPEC_ID)

    if cfg.model == "large-v3":
        automodel_kwargs = {"model": "Whisper-large-v3", "hub": "openai"}
    else:
        model_path = ensure(cfg.model)
        automodel_kwargs = {"model": str(model_path)}

    logger.info("Loading FunASR model %s on %s (+CAM++)", cfg.model, device)
    try:
        _funasr_pipeline = AutoModel(
            device=device,
            vad_model=str(vad_path),
            # speech_2_noise_ratio < default 1.0 down-weights speech vs noise so
            # BGM-filled pauses (audio-drama background music) score as silence and
            # VAD cuts there. Without it, a speaker change mid-segment over continuous
            # BGM stays one VAD segment → one CAM++ label (intra-segment conflation).
            # 0.7 splits the change, keeps speaker count stable, ~99% transcription
            # parity. Pairs with merge_vad=False above. See issue #384.
            vad_kwargs={
                "max_single_segment_time": 15000,
                "max_end_silence_time": 500,
                "speech_2_noise_ratio": 0.7,
            },
            spk_model=str(spk_path),
            disable_update=True,
            disable_pbar=True,
            **automodel_kwargs,
        )
    except Exception:
        logger.exception(
            "Failed to load ASR model %s on %s — check device compatibility and available memory",
            cfg.model,
            device,
        )
        raise
    _funasr_key = key
    return _funasr_pipeline


def _coerce_seconds(value: float) -> float:
    # FunASR reports ms when spk model is active, seconds otherwise.
    # CAM++ is always loaded, so values are always ms.
    return value / 1000.0


def _transcribe_funasr(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    gen_kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "use_itn": True,
        # merge_vad glues adjacent VAD segments into ~merge_length_s windows
        # REGARDLESS of speaker, before CAM++ embeds. A multi-speaker window then
        # collapses to one speaker label (under-clustering). Disabling it keeps
        # VAD's silence-bounded segments — finer, speaker-aligned — with no ASR
        # quality loss (measured ~98% transcription parity). See issue #384.
        "merge_vad": False,
    }
    if cfg.language and cfg.language != "auto":
        gen_kwargs["language"] = cfg.language

    # Lock spans load + generate: the first concurrent burst would otherwise
    # each build the singleton (~3.8GB transient), and generate() mutates
    # shared state. Holding it across both serializes the whole shared-model
    # critical section.
    try:
        with _transcribe_lock:
            model = _load_funasr(cfg)
            res = model.generate(**gen_kwargs)
    except Exception:
        logger.exception("FunASR generate failed for %s (model=%s)", audio_path, cfg.model)
        raise
    if not res:
        logger.warning("FunASR returned no results for %s", audio_path)
        return [], None

    first = res[0]
    raw = first.get("sentence_info") or first.get("segments") or []
    if not raw:
        logger.warning("FunASR returned result with no segments for %s: %s", audio_path, list(first.keys()))
        return [], None
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


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    """Transcribe audio. Returns (segments, detected_language).

    Segments carry speaker labels inline from FunASR's CAM++ diarization.
    """
    try:
        get_spec(cfg.model)  # raises ValueError on unknown model
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    return _transcribe_funasr(audio_path, cfg)


def build_speaker_namespace(segments: list[WhisperSegment]) -> dict[str | None, int]:
    """Map each distinct speaker in ``segments`` to an ordinal in first-seen order.

    Used to namespace speaker labels at render (``SPK{k}``). CAM++ labels are
    source-local; the ordinal + citation index (``S{N}·SPK{k}``) makes
    cross-source speaker conflation structurally impossible (spec Layer 5).

    The ordinal spans only the segments passed in. At chat time that is one
    turn's retrieved ranges, so ``SPK{k}`` is a per-turn render label, not a
    durable per-source speaker id.
    """
    ns: dict[str | None, int] = {}
    for seg in segments:
        if seg.speaker not in ns:
            ns[seg.speaker] = len(ns)
    return ns


def format_turns(
    segments: list[WhisperSegment],
    *,
    include_time: bool = False,
    citation_index: int | None = None,
    speaker_namespace: dict[str | None, int] | None = None,
) -> str:
    """Group consecutive same-speaker segments into speaker-turn lines.

    Shared by chat top-k reconstruction (``include_time`` + ``citation_index`` +
    ``speaker_namespace`` → ``[S{N}·SPK{k} @M:SS] text``), the UI viewer
    (``include_time``, raw label → ``[SPK_0 @M:SS] text``) and digest (neither →
    ``[SPK_0] text``). One helper, three variants (spec "Turn-text formatter").

    Time is ``@M:SS`` under an hour, ``@H:MM:SS`` at or past it (hours hidden when zero).
    """
    lines: list[str] = []
    i, n = 0, len(segments)
    while i < n:
        spk = segments[i].speaker
        start = segments[i].start
        texts = [segments[i].text]
        i += 1
        while i < n and segments[i].speaker == spk:
            texts.append(segments[i].text)
            i += 1
        if citation_index is not None and speaker_namespace is not None:
            label = f"S{citation_index}·SPK{speaker_namespace.get(spk, 0)}"
        else:
            label = spk or "SPK?"
        if include_time:
            h, rem = divmod(int(start), 3600)
            m, s = divmod(rem, 60)
            time = f" @{h}:{m:02d}:{s:02d}" if h else f" @{m}:{s:02d}"
        else:
            time = ""
        lines.append(f"[{label}{time}] {' '.join(texts)}")
    return "\n".join(lines)


async def load_transcript_text(source_id: str, *, include_time: bool = True) -> str:
    """Load a source's transcript from the segments table as speaker turns.

    Default (``include_time=True``) is the UI viewer view (turns + time, raw
    label). Digest/overview callers pass ``include_time=False`` (turns only).
    """
    from bibilab.db import get_transcript_segments  # local import avoids db<->pipeline cycle

    try:
        rows = await get_transcript_segments(source_id)
    except Exception:
        logger.exception("Failed to load transcript segments for source %s", source_id)
        raise
    segs = [WhisperSegment(start=r["start_s"], end=r["end_s"], text=r["text"], speaker=r["speaker"]) for r in rows]
    return format_turns(segs, include_time=include_time)


__all__ = [
    "WhisperSegment",
    "build_speaker_namespace",
    "format_turns",
    "load_transcript_text",
    "transcribe",
]
