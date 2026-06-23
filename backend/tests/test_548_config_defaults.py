"""Tests for #548: download stage config knobs (default concurrency + segments)."""

from bibilab.config import BibilabConfig


class TestBackendDownloadDefaults:
    """AC5 + AC6: #548 reverses the per-IP serialization premise; raise
    max_concurrent_downloads default to 2 and add download_segments knob default 4."""

    def test_max_concurrent_downloads_default_is_two(self) -> None:
        """Default 2 was 1 in #547 — premises falsified, parallel is now throughput-positive."""
        cfg = BibilabConfig()
        assert cfg.backend.max_concurrent_downloads == 2

    def test_download_segments_default_is_four(self) -> None:
        """Default 4 segments is the measured knee (issue: 4 segments ≈ 10× over single-stream)."""
        cfg = BibilabConfig()
        assert cfg.backend.download_segments == 4