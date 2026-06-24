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
        (("backend",), "max_concurrent_downloads"),
    ],
)
def test_removed_field_absent(path: tuple[str, ...], field: str) -> None:
    cfg = BibilabConfig()
    obj = cfg
    for attr in path:
        obj = getattr(obj, attr)
    assert field not in obj.model_dump()


def test_backend_download_connections_default() -> None:
    """Default per-file connection count fed to aria2c -x/-s."""
    assert BibilabConfig().backend.download_connections == 16


@pytest.mark.parametrize("bad", [0, -1, 65, 256])
def test_backend_download_connections_rejects_out_of_range(bad: int) -> None:
    """0 is meaningless to aria2c; >64 hits the same per-IP throttle this
    knob is meant to bound."""
    from pydantic import ValidationError

    from bibilab.config import BackendConfig

    with pytest.raises(ValidationError):
        BackendConfig(download_connections=bad)


def test_backend_download_connections_scales_down_at_high_jobs() -> None:
    """jobs=8 with default conns=16 would yield 128 sub-conns; auto-scale to 64//8=8."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=8)
    assert cfg.download_connections == 8
    assert cfg.max_concurrent_jobs * cfg.download_connections == 64


def test_backend_download_connections_floor_at_one() -> None:
    """Extreme jobs counts floor conns to 1 (aria2c -x1 still meaningful)."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=128)
    assert cfg.download_connections == 1
