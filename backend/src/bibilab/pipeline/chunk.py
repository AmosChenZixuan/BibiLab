"""Greedy segment merger — produces RAG-ready chunks from Whisper segments."""

from dataclasses import dataclass

import tiktoken

from bibilab.pipeline.transcribe import WhisperSegment

_enc = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 300
MAX_TOKENS = 400


@dataclass
class RagChunk:
    text: str
    timestamp_start: float
    timestamp_end: float
    sequence_index: int


def chunk_segments(
    segments: list[WhisperSegment],
    target_tokens: int = TARGET_TOKENS,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    buf_segs: list[WhisperSegment] = []
    buf_tokens = 0

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

        if seg_tokens >= MAX_TOKENS:
            # Oversized segment — flush current buffer first, then emit as its own chunk
            if buf_segs:
                flush(chunk_idx)
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

        if buf_tokens + seg_tokens > target_tokens and buf_segs:
            flush(chunk_idx)
            chunk_idx += 1
            buf_segs, buf_tokens = [], 0

        buf_segs.append(seg)
        buf_tokens += seg_tokens

    flush(chunk_idx)
    return chunks
