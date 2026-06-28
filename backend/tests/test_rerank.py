"""Tests for bibilab.pipeline.rerank."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_cross_encoder_loads_dir_returned_by_ensure(tmp_path: Path):
    """rerank must load the model from the dir ensure() returns for the
    config-selected spec — not a separately recomputed path. Otherwise a
    local_subdir change downloads to the new dir while the session reads the
    stale one (the bug #567 fixes)."""
    fake_dir = tmp_path / "reranker" / "Xenova_bge-reranker-base-q"
    fake_dir.mkdir(parents=True)

    captured: dict[str, str] = {}

    class _FakeSession:
        def __init__(self, path, providers=None, sess_options=None):
            captured["path"] = path
            captured["providers"] = providers

        def get_inputs(self):
            return []

    cfg = SimpleNamespace(rag=SimpleNamespace(reranker_spec_id="bge-reranker-base-q"))

    with (
        patch("bibilab.pipeline.rerank.load_config", return_value=cfg),
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

    mock_ensure.assert_called_once_with("bge-reranker-base-q")
    assert captured["path"] == str(fake_dir / "model.onnx")
    assert captured["providers"] == ["CPUExecutionProvider"]
