"""Speaker diarization via FunASR CAM++. Engine-agnostic pre-processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
    from funasr import AutoModel  # noqa: PLC0415

    if _pipeline is None or _pipeline_device != device:
        actual_device = "cuda:0" if device == "cuda" else "cpu"
        logger.info("Loading diarization model (CAM++) on %s", actual_device)
        _pipeline = AutoModel(
            model=None,
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=actual_device,
        )
        _pipeline_device = device
    return _pipeline


def diarize(audio_path: Path, device: str) -> list[SpeakerSegment]:
    """Run VAD + speaker diarization. Returns speaker segments sorted by time."""
    model = _load_diarization(device)
    res = model.generate(input=str(audio_path))
    if not res:
        return []
    first = res[0]
    raw_segs = first.get("sentence_info") or first.get("segments") or []
    segments = [
        SpeakerSegment(
            start=s["start"],
            end=s["end"],
            speaker=f"SPK_{s.get('spk', s.get('speaker', '?'))}",
        )
        for s in raw_segs
    ]
    segments.sort(key=lambda s: s.start)
    return segments
