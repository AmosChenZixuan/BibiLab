"""SenseVoice ASR engine via FunASR."""

from __future__ import annotations

import logging
from pathlib import Path

from bibilab.config import TranscriptionConfig

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_key: tuple[str, str] | None = None  # (model_size, device)

SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"


def _load_sensevoice(cfg: TranscriptionConfig):
    global _pipeline, _pipeline_key
    from funasr import AutoModel  # noqa: PLC0415

    key = (cfg.model_size, cfg.device)
    if _pipeline is None or _pipeline_key != key:
        device = "cuda:0" if cfg.device == "cuda" else "cpu"
        logger.info(
            "Loading SenseVoice model %s on %s from %s",
            cfg.model_size,
            device,
            SENSEVOICE_MODEL_ID,
        )
        _pipeline = AutoModel(
            model=SENSEVOICE_MODEL_ID,
            device=device,
            disable_punc=False,
        )
        _pipeline_key = key
    return _pipeline


def _transcribe_sensevoice(audio_path: Path, cfg: TranscriptionConfig) -> tuple[list, str | None]:
    """Transcribe audio with SenseVoice. Returns (WhisperSegment list, language)."""
    from bibilab.pipeline.transcribe import WhisperSegment

    model = _load_sensevoice(cfg)
    res = model.generate(
        input=str(audio_path),
        language=cfg.language,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
    )
    if not res:
        return [], None
    first = res[0]
    segments = [
        WhisperSegment(start=s["start"], end=s["end"], text=s["text"].strip()) for s in first.get("segments", [])
    ]
    lang = first.get("language")
    if lang == "auto":
        lang = None
    return segments, lang
