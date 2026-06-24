"""Regression tests for fields removed from BibilabConfig.

Each parametrized case asserts a removed field is absent from a default
BibilabConfig instance, so a future re-introduction is caught.
"""

import pytest

from bibilab.config import BibilabConfig


@pytest.mark.parametrize(
    "path,field",
    [
        ((), "vision"),
        (("accounts", "bilibili"), "last_verified"),
        (("ai",), "transcript_char_limit"),
        (("rag",), "chunk_pause_threshold"),
        ((), "transcript_collection_name"),
        (("transcription",), "llm_timeout"),
    ],
)
def test_removed_field_absent(path: tuple[str, ...], field: str) -> None:
    cfg = BibilabConfig()
    obj = cfg
    for attr in path:
        obj = getattr(obj, attr)
    assert field not in obj.model_dump()


def test_backend_download_connections_default() -> None:
    """Default per-file connection count fed to aria2c -x/-s.
    16 is calibrated: bounds the throttle tail AND matches the bench headline
    (max 5.7s vs native 60.3s on the throttled path)."""
    assert BibilabConfig().backend.download_connections == 16


@pytest.mark.parametrize("bad", [0, -1, 65, 256])
def test_backend_download_connections_rejects_out_of_range(bad: int) -> None:
    """0 is meaningless to aria2c; >64 hits the same per-IP throttle this
    knob is meant to bound."""
    from pydantic import ValidationError

    from bibilab.config import BackendConfig

    with pytest.raises(ValidationError):
        BackendConfig(download_connections=bad)
