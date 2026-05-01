"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from bibilab.config import models_dir
from bibilab.pipeline.embed import RetrievedChunk

logger = logging.getLogger(__name__)

# bge-reranker-base (XLM-RoBERTa) handles Chinese + English, matching
# the project's primary content languages. Model is intentionally fixed
# rather than configurable — swapping models would require re-tuning
# rerank_min_score since score distributions differ.
_MODEL_REPO = "Xenova/bge-reranker-base"
_MODEL_FILENAME = "model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"

_reranker: ONNXCrossEncoder | None = None
_reranker_lock = threading.Lock()


def _model_dir() -> Path:
    return models_dir("reranker", _MODEL_REPO.replace("/", "_"))


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
        onnx_input_names = {i.name for i in self._session.get_inputs()}
        has_token_type = "token_type_ids" in onnx_input_names

        encoded_list = [self._tokenizer.encode(query, doc) for query, doc in pairs]

        max_len = max(len(e.ids) for e in encoded_list)
        pad_id = self._tokenizer.token_to_id("<pad>") or 0

        batch_ids = []
        batch_mask = []
        batch_type_ids = [] if has_token_type else None

        for enc in encoded_list:
            pad_len = max_len - len(enc.ids)
            batch_ids.append(enc.ids + [pad_id] * pad_len)
            batch_mask.append(enc.attention_mask + [0] * pad_len)
            if has_token_type:
                batch_type_ids.append(enc.type_ids + [0] * pad_len)

        onnx_input = {
            "input_ids": self._np.array(batch_ids, dtype=self._np.int64),
            "attention_mask": self._np.array(batch_mask, dtype=self._np.int64),
        }
        if has_token_type:
            onnx_input["token_type_ids"] = self._np.array(batch_type_ids, dtype=self._np.int64)

        logits = self._session.run(None, onnx_input)[0]
        return [float(logits[i][0]) for i in range(len(pairs))]


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
