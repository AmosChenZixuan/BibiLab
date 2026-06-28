"""Unit tests for pipeline._shared helpers."""

import onnxruntime as ort

from bibilab.pipeline._shared import interpreting_providers


def test_interpreting_providers_drops_compiling_eps(monkeypatch):
    # CoreML (compiling) must be filtered; CUDA + CPU (interpreting) kept in order.
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert interpreting_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_interpreting_providers_macos_falls_back_to_cpu(monkeypatch):
    # The real macOS provider set: only CPU survives the whitelist.
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
