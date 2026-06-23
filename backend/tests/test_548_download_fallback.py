"""Tests for #548: native yt-dlp fallback when info dict signals fragmented
or non-https protocol. Pypdl segmented path must NOT run in these cases."""

from unittest.mock import MagicMock, patch

import pytest

from bibilab.adapters.bilibili import BilibiliAdapter


def _info(protocol: str = "https", fragments: list | None = None, ext: str = "m4a") -> dict:
    """Build an yt-dlp info dict as returned by extract_info(download=False)."""
    return {
        "url": "https://example.com/audio",
        "ext": ext,
        "protocol": protocol,
        "fragments": fragments,
        "http_headers": {"User-Agent": "yt-dlp/test"},
    }


@pytest.fixture
def captured_ydl():
    """Capture every yt_dlp.YoutubeDL instance + the args passed to extract_info."""
    instances: list = []

    class FakeYDL:
        def __init__(self, opts):
            instances.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return _info(protocol="mhtml_dash", fragments=[{"path": "seg0"}])

    return instances, FakeYDL


class TestDownloadFallback:
    """AC2: when info dict has non-https protocol OR fragments, segmented path
    is skipped and the existing native yt-dlp download is used (no regression)."""

    def test_fragmented_protocol_uses_native_ytdlp(self, tmp_path, captured_ydl):
        """protocol=mhtml_dash (or any non-https) must skip Pypdl entirely."""
        opts_list, FakeYDL = captured_ydl
        adapter = BilibiliAdapter(cookie="c")

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", FakeYDL):
            with patch("bibilab.adapters.bilibili.Pypdl") as mock_pypdl:
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter.download("BVfallback", "https://www.bilibili.com/video/BVfallback")

        # Pypdl must NEVER be instantiated on the fragmented path.
        mock_pypdl.assert_not_called()
        # Native yt-dlp ran twice: once for resolve (download=False), once for actual download.
        assert len(opts_list) == 2
        # Second invocation is the native fallback download.
        # We just check it had download-related yt-dlp retry knobs applied.
        assert "retries" in opts_list[1]

    def test_unknown_protocol_uses_native_ytdlp(self, tmp_path):
        """When yt-dlp returns a protocol pypdl can't handle, native path runs."""

        class FakeYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=False):
                return _info(protocol="rtmp")

        adapter = BilibiliAdapter(cookie="c")
        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", FakeYDL):
            with patch("bibilab.adapters.bilibili.Pypdl") as mock_pypdl:
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter.download("BVrtmp", "https://example.com/video")

        mock_pypdl.assert_not_called()