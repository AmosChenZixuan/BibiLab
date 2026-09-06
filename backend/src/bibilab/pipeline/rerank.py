"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import asyncio
import logging
import threading

from bibilab.model_registry import RERANKER_SPEC_ID, ensure
from bibilab.pipeline._shared import PAIR_WINDOW_TOKENS, QUERY_TOKEN_CLAMP, interpreting_providers
from bibilab.pipeline.chat_inference_pool import get_chat_pool
from bibilab.pipeline.embed import RetrievedChunk

logger = logging.getLogger(__name__)

# Cross-encoder is bge-reranker-base (XLM-RoBERTa) — Chinese + English, the
# project's primary content languages. One spec ships: the int8 quantized
# RERANKER_SPEC_ID. Quantization changes the cross-encoder's exact scores, but
# the model is deterministic on a kernel EP (CPU here), so the gateless top-k
# ordering stays reproducible per deployment.
_MODEL_FILENAME = "model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"

_reranker: ONNXCrossEncoder | None = None
_reranker_lock = threading.Lock()


def _clamp_query(length_tokenizer, query: str) -> str:
    """Pre-clamp the query to QUERY_TOKEN_CLAMP tokens before pair-encoding.

    `only_second` truncation only ever trims the document side, so without this
    a query longer than the clamp (but still under the full pair window) would
    silently eat into the document's guaranteed floor.

    `length_tokenizer` must have no truncation config: encoding this query alone
    on the pair tokenizer (which has `only_second` enabled) raises once the
    query alone is long enough to need truncating, since `only_second` requires
    a second sequence to trim. A dedicated untruncated instance also avoids
    toggling shared truncation state on the pair tokenizer, which `predict()`
    calls concurrently from a multi-worker thread pool.
    """
    ids = length_tokenizer.encode(query, add_special_tokens=False).ids
    if len(ids) <= QUERY_TOKEN_CLAMP:
        return query
    return length_tokenizer.decode(ids[:QUERY_TOKEN_CLAMP])


class ONNXCrossEncoder:
    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        self._np = np

        model_dir = ensure(RERANKER_SPEC_ID)
        import onnxruntime as ort  # noqa: PLC0415

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        # Providers from the shared helper — excludes compiler-based EPs (CoreML)
        # that OOM/hang this model on macOS; see interpreting_providers() for why.
        self._session = ort.InferenceSession(
            str(model_dir / _MODEL_FILENAME),
            providers=interpreting_providers(),
            sess_options=so,
        )
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tokenizer = Tokenizer.from_file(str(model_dir / _TOKENIZER_FILENAME))
        self._tokenizer.enable_truncation(max_length=PAIR_WINDOW_TOKENS, strategy="only_second")
        # Separate untruncated instance for _clamp_query's length check — see its
        # docstring for why it can't share self._tokenizer.
        self._length_tokenizer = Tokenizer.from_file(str(model_dir / _TOKENIZER_FILENAME))

    def predict(self, pairs: list[list[str]]) -> list[float]:
        onnx_input_names = {i.name for i in self._session.get_inputs()}
        has_token_type = "token_type_ids" in onnx_input_names

        encoded_list = [
            self._tokenizer.encode(_clamp_query(self._length_tokenizer, query), doc) for query, doc in pairs
        ]

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

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(get_chat_pool(), reranker.predict, pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    for chunk, score in scored:
        chunk.score = float(score)

    return [chunk for chunk, _ in scored[:top_k]]
