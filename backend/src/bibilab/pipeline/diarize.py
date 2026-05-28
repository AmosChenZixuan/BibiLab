"""Speaker diarization via FunASR CAM++. Engine-agnostic pre-processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from bibilab.asr_models import CAMPP_MODEL_ID, VAD_MODEL_ID
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_device: str | None = None


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


def _load_diarization(device: str):
    global _pipeline, _pipeline_device
    try:
        from funasr import AutoModel  # noqa: PLC0415
    except ImportError as exc:
        raise PipelineError(
            "Diarization requested but funasr is not installed. Run: uv sync --extra sensevoice"
        ) from exc

    if _pipeline is None or _pipeline_device != device:
        actual_device = "cuda:0" if device == "cuda" else "cpu"
        logger.info("Loading diarization model (CAM++) on %s", actual_device)
        _pipeline = AutoModel(
            model=None,
            vad_model=VAD_MODEL_ID,
            spk_model=CAMPP_MODEL_ID,
            device=actual_device,
        )
        _pipeline_device = device
    return _pipeline


def diarize(audio_path: Path, device: str) -> list[SpeakerSegment]:
    """Run VAD + speaker diarization. Returns speaker segments sorted by time."""
    model = _load_diarization(device)
    res = model.generate(input=str(audio_path))
    if not res:
        logger.warning("Diarization returned no results for %s", audio_path)
        return []
    first = res[0]
    raw_segs = first.get("sentence_info") or first.get("segments") or []
    if not raw_segs:
        logger.warning("Diarization produced no speaker segments for %s", audio_path)
        return []
    segments = [
        SpeakerSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker=f"SPK_{s.get('spk', s.get('speaker', '?'))}",
        )
        for s in raw_segs
    ]
    segments.sort(key=lambda s: s.start)
    return segments
