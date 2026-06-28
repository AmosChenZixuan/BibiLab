"""Tests for bibilab.pipeline.rerank."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_cross_encoder_loads_host_derived_spec_with_interpreting_providers(tmp_path: Path):
    """The session must load the dir ensure() returns for the host-derived spec
    and bind interpreting_providers() — never ort.get_available_providers(),
    which would let a compiling EP (CoreML/TensorRT) back in."""
    fake_dir = tmp_path / "reranker" / "Xenova_bge-reranker-base"
    fake_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, path, providers=None, sess_options=None):
            captured["path"] = path
            captured["providers"] = providers

        def get_inputs(self):
            return []

    with (
        patch("bibilab.pipeline.rerank.reranker_spec_id", return_value="bge-reranker-base"),
        patch("bibilab.pipeline.rerank.ensure", return_value=fake_dir) as mock_ensure,
        patch(
            "bibilab.pipeline.rerank.interpreting_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        patch("onnxruntime.InferenceSession", _FakeSession),
        patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()),
    ):
        from bibilab.pipeline.rerank import ONNXCrossEncoder

        ONNXCrossEncoder()

    mock_ensure.assert_called_once_with("bge-reranker-base")
    assert captured["path"] == str(fake_dir / "model.onnx")
    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
