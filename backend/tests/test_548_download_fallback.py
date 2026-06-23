"""Tests for native yt-dlp fallback when the resolve path can't hand off to pypdl.

Non-https / fragmented formats must skip the segmented path so existing yt-dlp
behaviors (which already handle them) are preserved.
"""

from unittest.mock import patch

import pytest

from bibilab.adapters.bilibili import BilibiliAdapter


class _FakeYDL:
    """Mock yt_dlp.YoutubeDL: capture opts, return a given info dict on resolve."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        return self._info

    @property
    def _info(self):
        # Hooked per-test by setting self.info before calling download().
        raise RuntimeError("test must set fake_ydl.info")


@pytest.fixture
def fake_ydl():
    captured: list = []

    def factory(opts):
        inst = _FakeYDL(opts)
        captured.append(inst)
        return inst

    return captured, factory


def test_fragmented_protocol_uses_native_ytdlp(tmp_path, fake_ydl) -> None:
    captured, factory = fake_ydl

    def resolve_side_effect(self, url, download=False):
        if download:
            return {"ext": "m4a"}
        self.info = {"protocol": "mhtml_dash", "fragments": [{"path": "seg0"}], "ext": "m4a"}
        return self.info

    _FakeYDL.extract_info = resolve_side_effect

    adapter = BilibiliAdapter(cookie="c")
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", factory):
        with patch("bibilab.adapters.bilibili.Pypdl") as mock_pypdl:
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BVfallback", "https://www.bilibili.com/video/BVfallback")

    mock_pypdl.assert_not_called()
    # Two yt-dlp invocations: resolve (download=False), native download (True).
    assert len(captured) == 2
    assert "retries" in captured[1].opts


def test_unknown_protocol_uses_native_ytdlp(tmp_path) -> None:
    """When yt-dlp returns a protocol pypdl can't handle, native path runs."""

    class FixedYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return {"protocol": "rtmp", "fragments": None, "ext": "m4a"}

    adapter = BilibiliAdapter(cookie="c")
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", FixedYDL):
        with patch("bibilab.adapters.bilibili.Pypdl") as mock_pypdl:
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BVrtmp", "https://example.com/video")

    mock_pypdl.assert_not_called()