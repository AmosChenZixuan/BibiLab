"""Tests for backend download config defaults (parallel + segmented paths)."""

from bibilab.config import BibilabConfig


class TestBackendDownloadDefaults:
    def test_max_concurrent_downloads_default_is_two(self) -> None:
        cfg = BibilabConfig()
        assert cfg.backend.max_concurrent_downloads == 2

    def test_download_segments_default_is_four(self) -> None:
        cfg = BibilabConfig()
        assert cfg.backend.download_segments == 4