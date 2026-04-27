"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from bibilab.pipeline.embed import RetrievedChunk

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None
_reranker_lock = threading.Lock()

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Rerank chunks using a cross-encoder model.

    Args:
        query: The user query.
        chunks: Chunks from initial retrieval.
        top_k: Number of top-scoring chunks to return.

    Returns:
        Top-k chunks sorted by cross-encoder score (most relevant first).
    """
    if not chunks:
        return []

    pairs = [[query, chunk.content] for chunk in chunks]
    reranker = _get_reranker()

    scores = await asyncio.to_thread(reranker.predict, pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    for chunk, score in scored[:top_k]:
        chunk.score = float(score)
        result.append(chunk)

    return result
