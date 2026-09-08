"""Greedy segment merger — produces RAG-ready chunks from punctuated sentence segments."""

import logging
from dataclasses import dataclass

from bibilab.pipeline._shared import DOC_TOKEN_BUDGET, count_tokens_xlmr
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

# Minimum fraction of DOC_TOKEN_BUDGET before a pause flush fires. Prevents
# flushing near-empty buffers on a pause; the token-forced flush below has no
# such guard — the ceiling is a hard invariant, so it flushes regardless of size.
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
    pause_threshold_seconds: float = 1.5,
) -> list[RagChunk]:
    min_flush_tokens = DOC_TOKEN_BUDGET * _MIN_TARGET_RATIO

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
        seg_tokens = count_tokens_xlmr(seg.text)

        if seg_tokens >= DOC_TOKEN_BUDGET:
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

        # Ceiling-forced flush. Merging seg into buf as-is would exceed
        # DOC_TOKEN_BUDGET, so free up room before appending it below. The
        # search never includes seg itself — a split has to land inside buf,
        # because any split point at or past buf's end embeds seg in the
        # flushed chunk, which by this branch's own guard already overflows.
        # Loops (not single-shot) because freeing room via one sentence split
        # can still leave a tail too big for seg — each iteration is a
        # sentence-bounded flush when one exists, else a forced cut; either
        # way buf_segs strictly shrinks, so the loop terminates. Afterward
        # buf_tokens + seg_tokens <= DOC_TOKEN_BUDGET always holds, so the
        # unconditional append below can never overshoot.
        while buf_segs and buf_tokens + seg_tokens > DOC_TOKEN_BUDGET:
            split_idx = _find_sentence_split(buf_segs, buf_seg_tokens, min_flush_tokens)
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
            else:
                # no boundary visible — forced cut, ceiling wins over avoiding a small chunk
                emit(chunk_idx, buf_segs, buf_seg_idxs)
                token_flushes += 1
                chunk_idx += 1
                buf_segs, buf_seg_idxs, buf_seg_tokens, buf_tokens = [], [], [], 0

        if buf_segs and buf_tokens + seg_tokens > DOC_TOKEN_BUDGET:
            # Loop invariant violated — would silently emit an over-ceiling chunk.
            raise ValueError(
                f"chunk_segments: buffer ({buf_tokens} tok) + segment ({seg_tokens} tok) "
                f"exceeds DOC_TOKEN_BUDGET ({DOC_TOKEN_BUDGET}) after the ceiling-forced flush"
            )
        buf_segs.append(seg)
        buf_seg_idxs.append(seg_i)
        buf_seg_tokens.append(seg_tokens)
        buf_tokens += seg_tokens

    emit(chunk_idx, buf_segs, buf_seg_idxs)

    logger.info(
        "chunk_segments: %d chunks from %d segments (ceiling=%d tokens)",
        len(chunks),
        len(segments),
        DOC_TOKEN_BUDGET,
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
