"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from bibilab.config import bibilab_home
from bibilab.pipeline.embed import RetrievedChunk

logger = logging.getLogger(__name__)

# Model is fixed to ensure consistent cross-encoder score semantics across deployments.
# Swapping models would require re-tuning the top_k rerank threshold since different
# models produce different score distributions.
_MODEL_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
_MODEL_FILENAME = "model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"

_reranker: ONNXCrossEncoder | None = None
_reranker_lock = threading.Lock()


def _model_dir() -> Path:
    return bibilab_home() / "models" / "reranker"


class ONNXCrossEncoder:
    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        self._np = np

        model_dir = _model_dir()
        self._ensure_downloaded(model_dir)
        import onnxruntime as ort  # noqa: PLC0415

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_dir / _MODEL_FILENAME),
            providers=ort.get_available_providers(),
            sess_options=so,
        )
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tokenizer = Tokenizer.from_file(str(model_dir / _TOKENIZER_FILENAME))
        self._tokenizer.enable_truncation(max_length=512)

    def _ensure_downloaded(self, model_dir: Path) -> None:
        model_path = model_dir / _MODEL_FILENAME
        tokenizer_path = model_dir / _TOKENIZER_FILENAME
        if model_path.exists() and tokenizer_path.exists():
            return
        model_dir.mkdir(parents=True, exist_ok=True)
        import httpx  # noqa: PLC0415

        base = f"https://huggingface.co/{_MODEL_REPO}/resolve/main"
        for remote_path, local_path in [
            ("onnx/" + _MODEL_FILENAME, model_path),
            (_TOKENIZER_FILENAME, tokenizer_path),
        ]:
            if local_path.exists():
                continue
            url = f"{base}/{remote_path}"
            logger.info("Downloading %s → %s", url, local_path)
            with httpx.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_bytes(1024 * 1024):
                        f.write(chunk)

    def predict(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            encoded = self._tokenizer.encode(query, doc)
            onnx_input = {
                "input_ids": self._np.array([encoded.ids], dtype=self._np.int64),
                "attention_mask": self._np.array([encoded.attention_mask], dtype=self._np.int64),
                "token_type_ids": self._np.array([encoded.type_ids], dtype=self._np.int64),
            }
            logits = self._session.run(None, onnx_input)[0]
            scores.append(float(logits[0][0]))
        return scores


def _get_reranker() -> ONNXCrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = ONNXCrossEncoder()
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

    for chunk, score in scored:
        chunk.score = float(score)

    return [chunk for chunk, _ in scored[:top_k]]
