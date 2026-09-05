"""Transcription stage.

All ASR models run on sherpa-onnx, CPU-only (see docs/decisions/0001-asr-runtime.md):
Silero VAD for segmentation, SenseVoice or Whisper (large-v3, int8) for recognition,
CAM++ speaker embeddings clustered over the same VAD spans ASR decodes.

Diarization is embedding + greedy clustering (_cluster_speakers), not sherpa-onnx's
own OfflineSpeakerDiarization — that class runs its own independent internal VAD,
which would decouple speaker regions from the ASR text segments they need to line
up with. This mirrors bench/asr/bench.py's Sherpa engine, the implementation that
measured the CER/speaker-agreement numbers behind this design.

ASR output is raw VAD-segment text (SenseVoice's own ITN, Whisper's own English
punctuation); the worker re-runs ct-punc on zh segments downstream
(see `pipeline/punctuate.py`) before persisting to `transcript_segments`.

VAD threshold/min_silence_duration (0.3 / 0.25) are a measured config — better
CER *and* speaker agreement than sherpa's own defaults (0.5 / 0.50) at once.
max_speech_duration=15s matches the pre-migration VAD chunk cap.

sherpa-onnx model assets auto-download on first ingest via model_registry.ensure(),
same pattern as the embedding/reranker models.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibilab.config import TranscriptionConfig
from bibilab.model_registry import (
    SHERPA_DIARIZATION_SPEC_ID,
    SHERPA_VAD_SPEC_ID,
    ensure,
    get_spec,
    resolve_transcription_spec_id,
)
from bibilab.pipeline._shared import format_hms, interpreting_provider
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# VAD is tuned for transcript quality, not speed: 0.3/0.25 measured ~11% slower than
# a laxer 0.5/0.5 and won it back in CER. int8 SenseVoice (pinned by the model spec)
# more than covers that cost — measured end to end at 69× realtime, c=4.
_SHERPA_NUM_THREADS = 4
_VAD_THRESHOLD = 0.3
_VAD_MIN_SILENCE_DURATION = 0.25
_VAD_MAX_SPEECH_DURATION = 15.0  # equivalent to the pre-migration 15s VAD chunk cap
_SPEAKER_CLUSTER_THRESHOLD = 0.5


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class _SherpaEngine:
    vad_cfg: Any
    recognizer: Any
    spk_extractor: Any


_sherpa_engine: _SherpaEngine | None = None
_sherpa_engine_key: tuple[str, str] | None = None  # (model, language)

# Guards singleton construction only, not inference. A shared sherpa model was
# measured safely serving concurrent workers with bit-identical output, so
# only the race to build the singleton needs serializing.
_transcribe_lock = threading.Lock()


def _load_sherpa(cfg: TranscriptionConfig) -> _SherpaEngine:
    global _sherpa_engine, _sherpa_engine_key

    # Both cfg.model and cfg.language are real construction-time inputs to the
    # recognizer (VAD/speaker never vary by either) — one cache, keyed on both,
    # reuses the existing single-keyed-singleton shape rather than splitting it.
    key = (cfg.model, cfg.language)
    with _transcribe_lock:
        if _sherpa_engine is not None and _sherpa_engine_key == key:
            return _sherpa_engine

        import sherpa_onnx as so  # noqa: PLC0415

        provider = interpreting_provider()

        vad_spec = get_spec(SHERPA_VAD_SPEC_ID)
        vad_dir = ensure(SHERPA_VAD_SPEC_ID)
        vad_cfg = so.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_dir / vad_spec.integrity_files[0])
        vad_cfg.silero_vad.threshold = _VAD_THRESHOLD
        vad_cfg.silero_vad.min_silence_duration = _VAD_MIN_SILENCE_DURATION
        vad_cfg.silero_vad.max_speech_duration = _VAD_MAX_SPEECH_DURATION
        vad_cfg.sample_rate = 16000
        vad_cfg.provider = provider
        vad_cfg.num_threads = _SHERPA_NUM_THREADS

        spec_id = resolve_transcription_spec_id(cfg.model)
        spec = get_spec(spec_id)
        model_dir = ensure(spec_id)
        if cfg.model == "large-v3":
            encoder, decoder, tokens = spec.integrity_files
            # language is fixed 'en' regardless of cfg.language: only int8 large-v3 on
            # English was validated (0.40 CER on Chinese — untested elsewhere).
            recognizer = so.OfflineRecognizer.from_whisper(
                encoder=str(model_dir / encoder),
                decoder=str(model_dir / decoder),
                tokens=str(model_dir / tokens),
                num_threads=_SHERPA_NUM_THREADS,
                provider=provider,
                language="en",
                task="transcribe",
            )
        else:
            model, tokens = spec.integrity_files
            recognizer = so.OfflineRecognizer.from_sense_voice(
                model=str(model_dir / model),
                tokens=str(model_dir / tokens),
                num_threads=_SHERPA_NUM_THREADS,
                provider=provider,
                language=cfg.language,
                use_itn=True,
            )

        spk_spec = get_spec(SHERPA_DIARIZATION_SPEC_ID)
        spk_dir = ensure(SHERPA_DIARIZATION_SPEC_ID)
        spk_extractor = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(
                model=str(spk_dir / spk_spec.integrity_files[0]),
                provider=provider,
                num_threads=_SHERPA_NUM_THREADS,
            )
        )

        _sherpa_engine = _SherpaEngine(vad_cfg=vad_cfg, recognizer=recognizer, spk_extractor=spk_extractor)
        _sherpa_engine_key = key
        return _sherpa_engine


def _vad_spans(vad_cfg: Any, samples: Any, rate: int) -> list[tuple[float, float]]:
    import sherpa_onnx as so  # noqa: PLC0415

    vad = so.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=100)
    window = vad_cfg.silero_vad.window_size
    spans: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            spans.append((seg.start / rate, (seg.start + len(seg.samples)) / rate))
            vad.pop()

    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i : i + window])
        drain()
    vad.flush()
    drain()
    return spans


def _cluster_speakers(embeddings: list) -> list[int]:
    """Greedy cosine agglomeration over an unknown speaker count. sherpa-onnx ships
    SpeakerEmbeddingManager for matching against *enrolled* speakers, not this —
    unsupervised clustering has no built-in sherpa-onnx equivalent."""
    import numpy as np  # noqa: PLC0415

    if not embeddings:
        return []
    embs = np.asarray(embeddings, dtype="float32")
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    centroids: list = []
    counts: list[int] = []
    labels: list[int] = []
    for e in embs:
        if centroids:
            sims = np.asarray(centroids) @ e
            best = int(sims.argmax())
            if sims[best] >= _SPEAKER_CLUSTER_THRESHOLD:
                labels.append(best)
                centroids[best] = (centroids[best] * counts[best] + e) / (counts[best] + 1)
                centroids[best] /= np.linalg.norm(centroids[best]) + 1e-9
                counts[best] += 1
                continue
        centroids.append(e.copy())
        counts.append(1)
        labels.append(len(centroids) - 1)
    return labels


def _transcribe_sherpa(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    import soundfile as sf  # noqa: PLC0415

    try:
        engine = _load_sherpa(cfg)
        # Everything below runs outside _transcribe_lock: concurrent decode/embed
        # calls against the same shared recognizer/extractor were measured to
        # produce bit-identical output, so no external serialization is needed
        # here (only building the singleton above needs the lock).
        samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        spans = _vad_spans(engine.vad_cfg, samples, rate)

        recognized: list[str] = []
        for start, end in spans:
            stream = engine.recognizer.create_stream()
            stream.accept_waveform(rate, samples[int(start * rate) : int(end * rate)])
            engine.recognizer.decode_stream(stream)
            recognized.append(stream.result.text.strip())

        embeddings = []
        for start, end in spans:
            stream = engine.spk_extractor.create_stream()
            stream.accept_waveform(rate, samples[int(start * rate) : int(end * rate)])
            stream.input_finished()
            embeddings.append(engine.spk_extractor.compute(stream))
        labels = _cluster_speakers(embeddings) if embeddings else []
    except Exception:
        logger.exception("sherpa-onnx transcription failed for %s (model=%s)", audio_path, cfg.model)
        raise

    segments: list[WhisperSegment] = []
    for (start, end), text, label in zip(spans, recognized, labels, strict=True):
        if not text:
            continue
        segments.append(WhisperSegment(start=start, end=end, text=text, speaker=f"SPK_{label}"))

    if not segments:
        logger.warning("sherpa-onnx returned no segments for %s", audio_path)
        return [], None

    # large-v3 wins because its recognizer is hard-coded to "en" (see _load_sherpa).
    if cfg.model == "large-v3":
        detected_lang = "en"
    else:
        detected_lang = cfg.language
    return segments, detected_lang


def transcribe(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list[WhisperSegment], str | None]:
    """Transcribe audio. Returns (segments, detected_language).

    Segments carry speaker labels from CAM++ embeddings clustered over the same
    VAD spans the ASR recognizer decodes.
    """
    try:
        get_spec(resolve_transcription_spec_id(cfg.model))  # raises ValueError on unknown model
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    return _transcribe_sherpa(audio_path, cfg)


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
            time = f" @{format_hms(start)}"
        else:
            time = ""
        lines.append(f"[{label}{time}] {' '.join(texts)}")
    return "\n".join(lines)


async def load_transcript_text(source_id: str, *, include_time: bool = True) -> str:
    """Load a source's transcript from the segments table as speaker turns.

    Default (``include_time=True``) is the UI viewer view (turns + time, raw
    label). Digest callers pass ``include_time=False`` (turns only).
    """
    from bibilab.db import get_transcript_segments, rows_to_segments  # local import avoids db<->pipeline cycle

    try:
        rows = await get_transcript_segments(source_id)
    except Exception:
        logger.exception("Failed to load transcript segments for source %s", source_id)
        raise
    return format_turns(rows_to_segments(rows), include_time=include_time)


__all__ = [
    "WhisperSegment",
    "build_speaker_namespace",
    "format_turns",
    "load_transcript_text",
    "transcribe",
]
