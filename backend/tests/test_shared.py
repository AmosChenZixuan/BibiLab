"""Unit tests for pipeline._shared helpers."""

import onnxruntime as ort

from bibilab.pipeline._shared import interpreting_providers


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
