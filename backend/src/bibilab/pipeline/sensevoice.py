"""SenseVoice ASR engine via FunASR."""

from __future__ import annotations

import logging
from pathlib import Path

from bibilab.asr_models import SENSEVOICE_MODEL_ID
from bibilab.config import TranscriptionConfig
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_key: tuple[str, str] | None = None  # (model_size, device)


def _load_sensevoice(cfg: TranscriptionConfig):
    global _pipeline, _pipeline_key
    try:
        from funasr import AutoModel  # noqa: PLC0415
    except ImportError as exc:
        raise PipelineError(
            "SenseVoice engine selected but funasr is not installed. Run: uv sync --extra sensevoice"
        ) from exc

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
        logger.warning("SenseVoice returned no results for %s", audio_path)
        return [], None
    first = res[0]
    raw = first.get("sentence_info") or first.get("segments") or []
    segments = [WhisperSegment(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip()) for s in raw]
    lang = first.get("language")
    if lang == "auto":
        lang = None
    return segments, lang
