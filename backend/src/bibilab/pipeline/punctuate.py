"""Standalone ct-punc punctuation + char-offset alignment.

Turns VAD `WhisperSegment`s into punctuated sentence `WhisperSegment`s for `zh`.
ct-punc runs OUTSIDE FunASR's AutoModel (decoupled) — it only inserts punctuation,
so a char-offset map back to the source segments recovers per-sentence speaker +
time exactly. Non-`zh` passes through unchanged (ASR punctuates other languages).
"""

from __future__ import annotations

import logging

from bibilab.pipeline.chunk import _SENT_END  # reuse — Code Health #3, correctness coupling
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

# Punctuation ct-punc may emit (stripped to build the alignment stream).
# Superset of _SENT_END (every terminal must also be strippable for the invariant).
_PUNC = frozenset("。，、！？；：．,.!?;:…　 ")
# Sentence-terminal set shared with the chunker (chunk._SENT_END): unambiguous
# ASCII !? split, ambiguous ASCII . / ; do not. Reused, never redefined.


def _strip_punc(s: str) -> str:
    return "".join(ch for ch in s if ch not in _PUNC)


def _align(segments: list[WhisperSegment], punctuated: str) -> list[WhisperSegment]:
    """Map a ct-punc output stream back onto source segments by char offset.

    Emits punctuated sentence segments. Splits on sentence-final punctuation and
    on speaker change (G3). Each segment's time is derived from the VAD segments
    its characters came from (G1: first.start..last.end; G2: shared span).

    Raises ValueError if the invariant `strip_punc(punctuated) == raw` is broken
    (ct-punc rewrote a character rather than only inserting punctuation).
    """
    # Per-source-char parallel arrays: speaker + index of the contributing VAD seg.
    char_spk: list[str | None] = []
    char_seg: list[int] = []
    for i, seg in enumerate(segments):
        for _ch in _strip_punc(seg.text):
            char_spk.append(seg.speaker)
            char_seg.append(i)
    raw = "".join(_strip_punc(s.text) for s in segments)

    if _strip_punc(punctuated) != raw:
        raise ValueError("ct-punc alignment invariant violated: output is not raw + inserted punctuation")

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
        if ch in _PUNC and ch not in _SENT_END:
            cur_text.append(ch)  # inserted comma / decimal point etc. — kept, does not advance raw
            continue
        if ch in _SENT_END:
            cur_text.append(ch)
            flush()
            continue
        # content char: advances the raw cursor
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
    flush()
    return out
