"""Standalone ct-punc punctuation + char-offset alignment.

Turns VAD `WhisperSegment`s into punctuated sentence `WhisperSegment`s for `zh`.
ct-punc runs OUTSIDE FunASR's AutoModel (decoupled) — it only inserts punctuation,
so a char-offset map back to the source segments recovers per-sentence speaker +
time exactly. Non-`zh` passes through unchanged (ASR punctuates other languages).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from bibilab.model_registry import PUNC_SPEC_ID, ensure
from bibilab.pipeline.chunk import _SENT_END  # reuse — Code Health #3, correctness coupling
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

# Punctuation ct-punc may emit (stripped to build the alignment stream).
# Superset of _SENT_END (every terminal must also be strippable for the invariant).
_PUNC = frozenset("。，、！？；：．,.!?;:…　 ")
assert all(ch in _PUNC for ch in _SENT_END), "_PUNC must be a superset of _SENT_END"
# Sentence-terminal set shared with the chunker (chunk._SENT_END): unambiguous
# ASCII !? split, ambiguous ASCII . / ; do not. Reused, never redefined.


def _strip_punc(s: str) -> str:
    return "".join(ch for ch in s if ch not in _PUNC)


def _align(segments: list[WhisperSegment], punctuated: str) -> list[WhisperSegment]:
    """Map a ct-punc output stream back onto source segments by char offset.

    Emits punctuated sentence segments. Splits on sentence-final punctuation and
    on speaker change (G3). Each segment's time is derived from the VAD segments
    its characters came from (first.start..last.end).

    Raises ValueError if source content is dropped or rewritten — i.e. the
    punctuated stream does not contain every raw char in order (ct-punc did more
    than insert punctuation). A char that does not match the raw cursor is
    treated as inserted punctuation regardless of which mark it is, so ct-punc
    may emit any punctuation outside `_PUNC` without tripping the invariant.
    """
    # Per-source-char parallel arrays: speaker + index of the contributing VAD seg.
    char_spk: list[str | None] = []
    char_seg: list[int] = []
    for i, seg in enumerate(segments):
        for _ch in _strip_punc(seg.text):
            char_spk.append(seg.speaker)
            char_seg.append(i)
    raw = "".join(_strip_punc(s.text) for s in segments)

    out: list[WhisperSegment] = []
    cur_text: list[str] = []
    cur_spk: str | None = None
    cur_segs: list[int] = []
    raw_i = 0

    def flush() -> None:
        nonlocal cur_text, cur_spk, cur_segs
        if cur_segs:
            first, last = segments[min(cur_segs)], segments[max(cur_segs)]
            out.append(
                WhisperSegment(
                    start=first.start,
                    end=last.end,
                    text="".join(cur_text),
                    speaker=cur_spk,
                )
            )
        cur_text, cur_spk, cur_segs = [], None, []

    for ch in punctuated:
        if raw_i < len(raw) and ch == raw[raw_i]:
            # content char: matches the raw cursor, advances it
            spk = char_spk[raw_i]
            seg_idx = char_seg[raw_i]
            raw_i += 1
            if cur_spk is None:
                cur_spk = spk
            elif spk != cur_spk:  # G3: speaker change forces a cut
                flush()
                cur_spk = spk
            cur_text.append(ch)
            cur_segs.append(seg_idx)
        else:
            # inserted punctuation (any mark) — kept, does not advance raw
            cur_text.append(ch)
            if ch in _SENT_END:
                flush()
    flush()

    if raw_i != len(raw):
        raise ValueError(
            f"ct-punc alignment invariant violated: consumed {raw_i}/{len(raw)} source chars "
            f"(model dropped or rewrote content, not pure punctuation insertion). "
            f"diverged near {raw[max(0, raw_i - 8) : raw_i + 8]!r}"
        )
    return out


# ---- ct-punc invocation -------------------------------------------------

_ctpunc_model: Any = None
_ctpunc_lock = threading.Lock()


def _run_ctpunc(raw: str) -> str:
    """Run standalone ct-punc on a raw (unpunctuated) char stream. Lazy singleton.

    Serialised via _ctpunc_lock: FunASR AutoModel.generate() mutates internal
    state (same class of bug as the ASR model — transcribe.py:47-51), so
    concurrent calls corrupt output.
    """
    global _ctpunc_model
    from funasr import AutoModel  # noqa: PLC0415

    if _ctpunc_model is None:
        with _ctpunc_lock:
            if _ctpunc_model is None:
                model_path = ensure(PUNC_SPEC_ID)
                logger.info("Loading ct-punc (standalone) from %s", model_path)
                _ctpunc_model = AutoModel(model=str(model_path), disable_update=True, disable_pbar=True)
    with _ctpunc_lock:
        try:
            return _ctpunc_model.generate(input=raw)[0]["text"]
        except Exception:
            logger.exception("ct-punc generate failed — resetting model for next attempt")
            _ctpunc_model = None
            raise


def punctuate(segments: list[WhisperSegment], language: str | None) -> list[WhisperSegment]:
    """Punctuate `zh` VAD segments into sentence segments; pass others through.

    Char-offset alignment strips spaces (correct for CJK, destructive for English),
    and the false-。 VAD defect is CJK-specific — so ct-punc is gated to `zh`.
    """
    if language != "zh" or not segments:
        return segments
    raw = "".join(_strip_punc(s.text) for s in segments)
    if not raw:
        return segments
    try:
        return _align(segments, _run_ctpunc(raw))
    except ValueError as exc:
        # Punctuation is an enhancement, never fatal: if ct-punc rewrote/dropped
        # content the alignment can't be trusted, so fall back to the raw VAD
        # segments (unpunctuated) and keep transcription succeeding.
        logger.warning("ct-punc alignment failed — using unpunctuated segments: %s", exc)
        return segments
