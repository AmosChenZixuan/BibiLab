"""Tests for bibilab.pipeline.rerank."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_cross_encoder_loads_dir_returned_by_ensure(tmp_path: Path):
    """rerank must load the model from the dir ensure() returns for the single
    RERANKER_SPEC_ID — not a separately recomputed path. Otherwise a local_subdir
    change downloads to the new dir while the session reads the stale one."""
    from bibilab.model_registry import RERANKER_SPEC_ID

    fake_dir = tmp_path / "reranker" / "Xenova_bge-reranker-base-q"
    fake_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, path, providers=None, sess_options=None):
            captured["path"] = path
            captured["providers"] = providers

        def get_inputs(self):
            return []

    with (
        patch("bibilab.pipeline.rerank.ensure", return_value=fake_dir) as mock_ensure,
        # The reranker must source providers from interpreting_providers() so
        # CoreML is excluded on macOS (the OOM/hang fix). Patch it to a sentinel
        # and assert that exact value reaches the session — proving the wiring,
        # not the EP list itself (that's covered in test_shared.py).
        patch(
            "bibilab.pipeline.rerank.interpreting_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("onnxruntime.InferenceSession", _FakeSession),
        patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()),
    ):
        from bibilab.pipeline.rerank import ONNXCrossEncoder

        ONNXCrossEncoder()

    mock_ensure.assert_called_once_with(RERANKER_SPEC_ID)
    assert captured["path"] == str(fake_dir / "model.onnx")
    assert captured["providers"] == ["CPUExecutionProvider"]
