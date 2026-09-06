"""Unit tests for ONNXMultilingualEmbedding — no real ONNX/Chroma involved."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_encoding(ids: list[int]):
    enc = MagicMock()
    enc.ids = ids
    enc.attention_mask = [1] * len(ids)
    enc.type_ids = [0] * len(ids)
    return enc


def _build_embedding(tmp_path: Path):
    """Construct ONNXMultilingualEmbedding with ensure/session/tokenizer mocked,
    returning the instance plus the tokenizer mock so tests can inspect .encode
    calls. Mirrors the ensure()/InferenceSession mocking pattern in test_rerank.py."""
    model_dir = tmp_path / "embedding" / "intfloat_multilingual-e5-small"
    (model_dir / "onnx").mkdir(parents=True)

    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: _fake_encoding([1, 2, 3])

    fake_session = MagicMock()
    fake_session.run.return_value = [[[0.1, 0.2] for _ in range(3)] for _ in range(1)]

    with (
        patch("bibilab.pipeline.embed.ensure", return_value=model_dir),
        patch("onnxruntime.InferenceSession", return_value=fake_session),
        patch("tokenizers.Tokenizer.from_file", return_value=tokenizer),
    ):
        from bibilab.pipeline.embed import ONNXMultilingualEmbedding

        instance = ONNXMultilingualEmbedding()

    return instance, tokenizer


def test_loads_model_from_dir_ensure_returns(tmp_path: Path):
    """The embedding model must load from the dir ensure() returns for
    EMBEDDING_SPEC_ID — not a separately recomputed path. Otherwise a
    local_subdir change (#715 moved it under embedding/intfloat_multilingual-e5-small
    so a stale MiniLM download can't masquerade as the new model) downloads to the
    new dir while the session loads from a stale hardcoded one. Mirrors
    test_rerank.py::test_cross_encoder_loads_dir_returned_by_ensure."""
    model_dir = tmp_path / "embedding" / "intfloat_multilingual-e5-small"
    (model_dir / "onnx").mkdir(parents=True)
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: _fake_encoding([1, 2, 3])

    with (
        patch("bibilab.pipeline.embed.ensure", return_value=model_dir) as mock_ensure,
        patch("onnxruntime.InferenceSession") as mock_session_cls,
        patch("tokenizers.Tokenizer.from_file", return_value=tokenizer),
    ):
        from bibilab.model_registry import EMBEDDING_SPEC_ID
        from bibilab.pipeline.embed import ONNXMultilingualEmbedding

        ONNXMultilingualEmbedding()

    mock_ensure.assert_called_once_with(EMBEDDING_SPEC_ID)
    assert mock_session_cls.call_args.args[0] == str(model_dir / "onnx" / "model.onnx")


def test_call_prefixes_passages(tmp_path: Path):
    """embed_chunks -> collection.add() reaches __call__; every text sent to the
    tokenizer must carry the model card's 'passage: ' prefix."""
    instance, tokenizer = _build_embedding(tmp_path)

    instance(["hello world"])

    tokenizer.encode.assert_called_once_with("passage: hello world")


def test_embed_query_prefixes_queries(tmp_path: Path):
    """query_chunks -> collection.query() reaches embed_query (Chroma dispatches
    query_texts to embed_query, not __call__); the query prefix must differ from
    the passage prefix."""
    instance, tokenizer = _build_embedding(tmp_path)

    instance.embed_query(["hello world"])

    tokenizer.encode.assert_called_once_with("query: hello world")


def test_passage_and_query_prefixes_are_not_swapped(tmp_path: Path):
    instance, tokenizer = _build_embedding(tmp_path)

    instance(["a"])
    instance.embed_query(["a"])

    calls = [c.args[0] for c in tokenizer.encode.call_args_list]
    assert calls == ["passage: a", "query: a"]
