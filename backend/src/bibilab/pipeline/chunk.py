"""Greedy segment merger — produces RAG-ready chunks from punctuated sentence segments."""

import logging
from dataclasses import dataclass

from bibilab.pipeline._shared import count_tokens
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

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

# Minimum fraction of target_tokens before a pause or token flush fires.
# Prevents flushing near-empty buffers.
_MIN_TARGET_RATIO = 0.5

# Sentence-ending punctuation — token flush splits on the latest occurrence.
# ASCII "." and ";" omitted: ambiguous on decimals, abbreviations, code, URLs,
# and list separators. "!" and "?" kept (unambiguous).
_SENT_END: tuple[str, ...] = ("。", "！", "？", "．", "…", "!", "?")


@dataclass
class RagChunk:
    text: str
    timestamp_start: float
    timestamp_end: float
    sequence_index: int
    seg_start: int
    seg_end: int

    def __post_init__(self):
        if self.seg_start < 0 or self.seg_end < self.seg_start:
            raise ValueError(f"Invalid seg range: [{self.seg_start}, {self.seg_end}]")


def _find_sentence_split(
    segs: list[WhisperSegment],
    seg_tokens: list[int],
    min_tokens: float,
) -> int | None:
    """Return the latest index i where segs[:i+1] ends at a sentence boundary
    and its token sum meets min_tokens. None if no qualifying boundary."""
    cum = 0
    last_idx: int | None = None
    for i, (s, t) in enumerate(zip(segs, seg_tokens)):
        cum += t
        if s.text.rstrip().endswith(_SENT_END) and cum >= min_tokens:
            last_idx = i
    return last_idx


def chunk_segments(
    segments: list[WhisperSegment],
    target_tokens: int | None = None,
    chunk_max_tokens: int | None = None,
    language: str = "en",
    pause_threshold_seconds: float = 1.5,
) -> list[RagChunk]:
    resolved_target = (
        target_tokens if target_tokens is not None else _LANG_TARGET_TOKENS.get(language, _DEFAULT_TARGET_TOKENS)
    )
    resolved_max = chunk_max_tokens if chunk_max_tokens is not None else int(resolved_target * _MAX_TOKEN_RATIO)
    min_flush_tokens = resolved_target * _MIN_TARGET_RATIO

    chunks: list[RagChunk] = []
    buf_segs: list[WhisperSegment] = []
    buf_seg_idxs: list[int] = []  # original segment indices for seg_start/seg_end
    buf_seg_tokens: list[int] = []
    buf_tokens = 0
    pause_flushes = 0
    token_flushes = 0
    sentence_flushes = 0
    oversized_flushes = 0

    def emit(idx: int, segs: list[WhisperSegment], idxs: list[int]) -> None:
        if not segs:
            return
        chunks.append(
            RagChunk(
                text=" ".join(s.text for s in segs),
                timestamp_start=segs[0].start,
                timestamp_end=segs[-1].end,
                sequence_index=idx,
                seg_start=idxs[0],
                seg_end=idxs[-1],
            )
        )

    chunk_idx = 0
    for seg_i, seg in enumerate(segments):
        seg_tokens = count_tokens(seg.text)

        if seg_tokens >= resolved_max:
            # Oversized segment — flush current buffer first, then emit as its own chunk
            if buf_segs:
                emit(chunk_idx, buf_segs, buf_seg_idxs)
                oversized_flushes += 1
                chunk_idx += 1
                buf_segs, buf_seg_idxs, buf_seg_tokens, buf_tokens = [], [], [], 0
            chunks.append(
                RagChunk(
                    text=seg.text,
                    timestamp_start=seg.start,
                    timestamp_end=seg.end,
                    sequence_index=chunk_idx,
                    seg_start=seg_i,
                    seg_end=seg_i,
                )
            )
            chunk_idx += 1
            continue

        # Pause-aware flush: if buffer has enough content and a long pause
        # precedes this segment, flush before merging across the boundary.
        if buf_segs:
            gap = seg.start - buf_segs[-1].end
            if gap > pause_threshold_seconds and buf_tokens >= min_flush_tokens:
                emit(chunk_idx, buf_segs, buf_seg_idxs)
                pause_flushes += 1
                chunk_idx += 1
                buf_segs, buf_seg_idxs, buf_seg_tokens, buf_tokens = [], [], [], 0

        # Token-target flush. Considers buf + incoming seg together. Prefers
        # the latest sentence boundary anywhere in that window; if none,
        # flushes the buffer at target to bound chunk size.
        if buf_tokens + seg_tokens > resolved_target and buf_segs:
            split_idx = _find_sentence_split(
                buf_segs + [seg],
                buf_seg_tokens + [seg_tokens],
                min_flush_tokens,
            )
            if split_idx == len(buf_segs):
                # boundary is on the incoming seg — flush buf + seg together
                emit(chunk_idx, buf_segs + [seg], buf_seg_idxs + [seg_i])
                sentence_flushes += 1
                chunk_idx += 1
                buf_segs, buf_seg_idxs, buf_seg_tokens, buf_tokens = [], [], [], 0
                continue  # seg already consumed
            if split_idx is not None:
                # boundary inside buf — flush head, keep tail
                head_count = split_idx + 1
                head_tokens = sum(buf_seg_tokens[:head_count])
                emit(chunk_idx, buf_segs[:head_count], buf_seg_idxs[:head_count])
                sentence_flushes += 1
                chunk_idx += 1
                buf_segs = buf_segs[head_count:]
                buf_seg_idxs = buf_seg_idxs[head_count:]
                buf_seg_tokens = buf_seg_tokens[head_count:]
                buf_tokens -= head_tokens
            elif buf_tokens >= min_flush_tokens:
                # no boundary visible — bound chunk size at target
                emit(chunk_idx, buf_segs, buf_seg_idxs)
                token_flushes += 1
                chunk_idx += 1
                buf_segs, buf_seg_idxs, buf_seg_tokens, buf_tokens = [], [], [], 0

        buf_segs.append(seg)
        buf_seg_idxs.append(seg_i)
        buf_seg_tokens.append(seg_tokens)
        buf_tokens += seg_tokens

    emit(chunk_idx, buf_segs, buf_seg_idxs)

    logger.info(
        "chunk_segments: %d chunks from %d segments (target=%d tokens)",
        len(chunks),
        len(segments),
        resolved_target,
    )
    # Only forced cuts are actionable: a token/oversized flush means no trustworthy
    # sentence boundary was available, so the chunk was cut mid-meaning. Pause and
    # sentence flushes are the healthy path and not worth reporting per run.
    forced = token_flushes + oversized_flushes
    if forced:
        total_flushes = pause_flushes + token_flushes + sentence_flushes + oversized_flushes
        logger.warning(
            "chunk_segments: %d of %d cuts fell back to a non-sentence boundary "
            "(token-forced=%d, oversized=%d) — punctuation may be sparse",
            forced,
            total_flushes,
            token_flushes,
            oversized_flushes,
        )

    return chunks
