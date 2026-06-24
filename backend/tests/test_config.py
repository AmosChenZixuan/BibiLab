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


def test_backend_download_connections_keeps_default_at_jobs_1() -> None:
    """Single-video (jobs=1) keeps the bench-calibrated 16 — the value prop
    the per-IP budget exists to protect, not flatten."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=1)
    assert cfg.download_connections == 16


def test_backend_download_connections_scales_down_at_high_jobs() -> None:
    """With max_concurrent_jobs=8 the user's stated 16 would yield 128
    sub-connections (>64 ceiling); auto-scale to 64//8=8."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=8, download_connections=16)
    assert cfg.download_connections == 8
    assert cfg.max_concurrent_jobs * cfg.download_connections == 64


def test_backend_download_connections_scales_at_jobs_4() -> None:
    """With max_concurrent_jobs=4 the user's stated 16 yields exactly 64,
    which is the ceiling — no scaling needed."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=4, download_connections=16)
    assert cfg.download_connections == 16


def test_backend_download_connections_floor_at_one() -> None:
    """Even at extreme jobs counts, download_connections never drops below 1
    (aria2c -x1 is still meaningful — it just doesn't add parallelism)."""
    from bibilab.config import BackendConfig

    cfg = BackendConfig(max_concurrent_jobs=128, download_connections=16)
    assert cfg.download_connections == 1
