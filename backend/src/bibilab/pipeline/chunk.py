"""Greedy segment merger — produces RAG-ready chunks from Whisper segments."""

import logging
from dataclasses import dataclass

import tiktoken

from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

# Token targets by language.  Chinese encodes at roughly 1 token per character
# in cl100k_base, while English encodes at roughly 1 token per 4 characters.
# To keep the semantic span comparable, Chinese gets a proportionally larger
# target.  Unknown languages fall back to the English default.
_DEFAULT_TARGET_TOKENS = 300
_LANG_TARGET_TOKENS: dict[str, int] = {
    "zh": 800,
    "en": _DEFAULT_TARGET_TOKENS,
}
_MAX_TOKEN_RATIO = 4 / 3

# Minimum fraction of target_tokens before a pause triggers a flush.
# Prevents flushing near-empty buffers on every short pause.
_MIN_TARGET_RATIO = 0.5

# Sentence-ending punctuation — token flush prefers these as boundaries.
# Covers CJK terminals (incl. ellipsis, full-width period/semicolon) and ASCII.
_SENT_END: tuple[str, ...] = ("。", "！", "？", "．", "；", "…", ".", "!", "?", ";")

# Fraction of resolved_max at which a token-overflow flush fires even
# without a sentence-ending boundary — bounds worst-case chunk size.
_NEAR_HARD_CAP_RATIO = 0.9


@dataclass
class RagChunk:
    text: str
    timestamp_start: float
    timestamp_end: float
    sequence_index: int


def chunk_segments(
    segments: list[WhisperSegment],
    target_tokens: int | None = None,
    chunk_max_tokens: int | None = None,
    language: str = "en",
    # Default must match RagConfig.chunk_pause_threshold in config.py.
    pause_threshold_seconds: float = 1.5,
) -> list[RagChunk]:
    resolved_target = (
        target_tokens if target_tokens is not None else _LANG_TARGET_TOKENS.get(language, _DEFAULT_TARGET_TOKENS)
    )
    resolved_max = chunk_max_tokens if chunk_max_tokens is not None else int(resolved_target * _MAX_TOKEN_RATIO)

    chunks: list[RagChunk] = []
    buf_segs: list[WhisperSegment] = []
    buf_tokens = 0
    pause_flushes = 0
    token_flushes = 0
    sentence_flushes = 0
    oversized_flushes = 0

    def flush(idx: int) -> None:
        if not buf_segs:
            return
        chunks.append(
            RagChunk(
                text=" ".join(s.text for s in buf_segs),
                timestamp_start=buf_segs[0].start,
                timestamp_end=buf_segs[-1].end,
                sequence_index=idx,
            )
        )

    chunk_idx = 0
    for seg in segments:
        seg_tokens = len(_enc.encode(seg.text))

        if seg_tokens >= resolved_max:
            # Oversized segment — flush current buffer first, then emit as its own chunk
            if buf_segs:
                flush(chunk_idx)
                oversized_flushes += 1
                chunk_idx += 1
                buf_segs, buf_tokens = [], 0
            chunks.append(
                RagChunk(
                    text=seg.text,
                    timestamp_start=seg.start,
                    timestamp_end=seg.end,
                    sequence_index=chunk_idx,
                )
            )
            chunk_idx += 1
            continue

        # Pause-aware flush: if buffer has enough content and a long pause
        # precedes this segment, flush before merging across the boundary.
        if buf_segs:
            gap = seg.start - buf_segs[-1].end
            if gap > pause_threshold_seconds and buf_tokens >= resolved_target * _MIN_TARGET_RATIO:
                flush(chunk_idx)
                pause_flushes += 1
                chunk_idx += 1
                buf_segs, buf_tokens = [], 0

        # Token-target flush: prefers sentence-ending punctuation as
        # boundary. Falls back to hard-cap flush near resolved_max to
        # bound worst-case chunk size when no sentence end appears.
        if buf_tokens + seg_tokens > resolved_target and buf_segs:
            last_text = buf_segs[-1].text.rstrip()
            ends_at_sentence = last_text.endswith(_SENT_END)
            near_hard_cap = buf_tokens + seg_tokens > resolved_max * _NEAR_HARD_CAP_RATIO

            if ends_at_sentence and buf_tokens >= resolved_target * _MIN_TARGET_RATIO:
                flush(chunk_idx)
                sentence_flushes += 1
                chunk_idx += 1
                buf_segs, buf_tokens = [], 0
            elif near_hard_cap and buf_tokens >= resolved_target * _MIN_TARGET_RATIO:
                flush(chunk_idx)
                token_flushes += 1
                chunk_idx += 1
                buf_segs, buf_tokens = [], 0

        buf_segs.append(seg)
        buf_tokens += seg_tokens

    flush(chunk_idx)

    total_flushes = pause_flushes + token_flushes + sentence_flushes + oversized_flushes
    if total_flushes:
        logger.info(
            "chunk_segments: %d chunks from %d segments (pause=%d, token=%d, sentence=%d, oversized=%d, target=%d)",
            len(chunks),
            len(segments),
            pause_flushes,
            token_flushes,
            sentence_flushes,
            oversized_flushes,
            resolved_target,
        )

    return chunks
