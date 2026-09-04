"""Unit tests for pipeline._shared helpers."""

import onnxruntime as ort

from bibilab.pipeline._shared import interpreting_provider, interpreting_providers


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
