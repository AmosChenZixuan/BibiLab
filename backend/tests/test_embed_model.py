"""Unit tests for ONNXMultilingualEmbedding — no real ONNX/Chroma involved."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _fake_encoding(ids: list[int]):
    enc = MagicMock()
    enc.ids = ids
    enc.attention_mask = [1] * len(ids)
    enc.type_ids = [0] * len(ids)
    return enc


def _build_embedding(tmp_path: Path):
    """Construct ONNXMultilingualEmbedding with ensure/session/tokenizer mocked.
    Mirrors test_rerank.py::test_cross_encoder_loads_dir_returned_by_ensure's
    mocking pattern, including its named-capture _FakeSession (over a bare
    MagicMock) and its interpreting_providers() sentinel-and-assert wiring
    check — the embedder sources providers from the same helper for the same
    CoreML-exclusion reason.

    Returns (instance, tokenizer_mock, captured, ensure_mock): captured holds
    the path/providers the session was constructed with; ensure_mock lets
    callers assert on EMBEDDING_SPEC_ID."""
    model_dir = tmp_path / "embedding" / "intfloat_multilingual-e5-small"
    (model_dir / "onnx").mkdir(parents=True)

    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: _fake_encoding([1, 2, 3])

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, path, providers=None, sess_options=None):
            captured["path"] = path
            captured["providers"] = providers

        def run(self, *args, **kwargs):
            return [[[0.1, 0.2] for _ in range(3)] for _ in range(1)]

    with (
        patch("bibilab.pipeline.embed.ensure", return_value=model_dir) as mock_ensure,
        patch(
            "bibilab.pipeline.embed.interpreting_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("onnxruntime.InferenceSession", _FakeSession),
        patch("tokenizers.Tokenizer.from_file", return_value=tokenizer),
    ):
        from bibilab.pipeline.embed import ONNXMultilingualEmbedding

        instance = ONNXMultilingualEmbedding()

    return instance, tokenizer, captured, mock_ensure


def test_loads_model_from_dir_ensure_returns(tmp_path: Path):
    """The embedding model must load from the dir ensure() returns for
    EMBEDDING_SPEC_ID — not a separately recomputed path. Otherwise a
    local_subdir change (this one moved it under
    embedding/intfloat_multilingual-e5-small so a stale MiniLM download can't
    masquerade as the new model) downloads to the new dir while the session
    loads from a stale hardcoded one. Mirrors
    test_rerank.py::test_cross_encoder_loads_dir_returned_by_ensure."""
    from bibilab.model_registry import EMBEDDING_SPEC_ID

    _, _, captured, mock_ensure = _build_embedding(tmp_path)
    model_dir = tmp_path / "embedding" / "intfloat_multilingual-e5-small"

    mock_ensure.assert_called_once_with(EMBEDDING_SPEC_ID)
    assert captured["path"] == str(model_dir / "onnx" / "model.onnx")
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_call_prefixes_passages(tmp_path: Path):
    """embed_chunks -> collection.add() reaches __call__; every text sent to the
    tokenizer must carry the model card's 'passage: ' prefix."""
    instance, tokenizer, _, _ = _build_embedding(tmp_path)

    instance(["hello world"])

    tokenizer.encode.assert_called_once_with("passage: hello world")


def test_embed_query_prefixes_queries(tmp_path: Path):
    """query_chunks -> collection.query() reaches embed_query (Chroma dispatches
    query_texts to embed_query, not __call__); the query prefix must differ from
    the passage prefix."""
    instance, tokenizer, _, _ = _build_embedding(tmp_path)

    instance.embed_query(["hello world"])

    tokenizer.encode.assert_called_once_with("query: hello world")


def test_embeddings_are_unit_norm(tmp_path: Path):
    """Vectors must leave the embedder L2-normalized, on both the passage and the
    query side. The collection is queried in Chroma's default L2 space, so an
    unnormalized vector ranks by magnitude rather than angle and lands far
    outside the [0, 2] range query_chunks' max_distance floor assumes — which
    silently drops every vector hit and degrades hybrid search to BM25-only."""
    instance, _, _, _ = _build_embedding(tmp_path)

    for vec in (instance(["hello world"])[0], instance.embed_query(["hello world"])[0]):
        assert sum(x * x for x in vec) == pytest.approx(1.0)
