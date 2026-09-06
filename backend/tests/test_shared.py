"""Unit tests for pipeline._shared helpers."""

from unittest.mock import patch

import onnxruntime as ort

import bibilab.pipeline._shared as shared
from bibilab.pipeline._shared import count_tokens_xlmr, interpreting_provider, interpreting_providers


def _toy_xlmr_tokenizer():
    """Real (non-mocked) tokenizers.Tokenizer, toy vocab — same hermetic
    pattern test_rerank.py uses to exercise real tokenizer library code
    without downloading the actual XLM-R model file."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"<unk>": 0}
    vocab.update({f"w{i}": i + 1 for i in range(200)})
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return tok


def test_count_tokens_xlmr_uses_cached_singleton(monkeypatch):
    """Once loaded, count_tokens_xlmr reuses the module-level singleton — same
    double-checked-lock pattern as rerank.py's _get_reranker()."""
    monkeypatch.setattr(shared, "_xlmr_tokenizer", _toy_xlmr_tokenizer())
    assert count_tokens_xlmr("w0 w1 w2") == 3


def test_count_tokens_xlmr_loads_tokenizer_file_via_ensure(tmp_path, monkeypatch):
    """First call loads onnx/tokenizer.json from ensure(EMBEDDING_SPEC_ID) — the
    same file embed.py's ONNXMultilingualEmbedding reads — with no ONNX
    session construction."""
    monkeypatch.setattr(shared, "_xlmr_tokenizer", None)
    model_dir = tmp_path / "embedding"
    (model_dir / "onnx").mkdir(parents=True)
    _toy_xlmr_tokenizer().save(str(model_dir / "onnx" / "tokenizer.json"))

    with patch("bibilab.model_registry.ensure", return_value=model_dir) as mock_ensure:
        count = count_tokens_xlmr("w0 w1 w2 w3")

    mock_ensure.assert_called_once()
    assert count == 4


def test_interpreting_providers_drops_compiling_eps(monkeypatch):
    # Compiler-based EPs (CoreML, DirectML) filtered; kernel-based EPs
    # (CUDA, ROCm, CPU) kept in priority order.
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: [
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert interpreting_providers() == [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_interpreting_providers_macos_falls_back_to_cpu(monkeypatch):
    # The real macOS provider set: only CPU survives the allowlist.
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: [
            "CoreMLExecutionProvider",
            "AzureExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert interpreting_providers() == ["CPUExecutionProvider"]


def test_interpreting_provider_translates_cpu_ep(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert interpreting_provider() == "cpu"


def test_interpreting_provider_translates_cuda_ep(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert interpreting_provider() == "cuda"


def test_interpreting_provider_unrecognized_ep_falls_back_to_cpu(monkeypatch):
    # ROCm is kernel-based and allowlisted by interpreting_providers(), but sherpa-onnx
    # has no distinct "rocm" provider string — fall back to cpu rather than guessing.
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["ROCMExecutionProvider"])
    assert interpreting_provider() == "cpu"


def test_interpreting_provider_empty_providers_falls_back_to_cpu(monkeypatch):
    # A slim/custom onnxruntime build could report no CUDA/ROCm/CPU provider at all —
    # must not crash with an IndexError on the first ASR call.
    monkeypatch.setattr(ort, "get_available_providers", lambda: [])
    assert interpreting_provider() == "cpu"
