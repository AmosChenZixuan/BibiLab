"""Tests for the segmented pypdl path on direct https bestaudio URLs.

Verifies that the resolved URL + headers (with cookie + Referer) reach pypdl
with the expected segment/retry budget, and that a short file is rejected.
"""

from unittest.mock import MagicMock, patch

import pytest

from bibilab.adapters.base import DownloadError
from bibilab.adapters.bilibili import BilibiliAdapter


def _https_info(url: str = "https://example.com/audio.m4s", headers: dict | None = None, filesize: int = 1234) -> dict:
    return {
        "url": url,
        "ext": "m4a",
        "protocol": "https",
        "fragments": None,
        "http_headers": headers or {"User-Agent": "yt-dlp/test"},
        "filesize": filesize,
    }


@pytest.fixture
def fake_pypdl():
    """Patch Pypdl + pre-create the fake output file so download() can succeed."""
    (None)  # placeholder; each test sets up its own


@pytest.fixture
def pypdl_setup(tmp_path):
    """Patch yt-dlp + pypdl and yield (Pypdl_mock_instance, tmp_path) with the
    expected output file pre-created so the size check passes by default."""

    def _make(video_id: str, *, cookie: str = "", info: dict | None = None, body: bytes | None = None) -> tuple:
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        out = tmp_path / "downloads" / f"{video_id}.m4a"
        out.write_bytes(body if body is not None else b"x" * 1234)

        info = info if info is not None else _https_info()
        pypdl_instance = MagicMock()

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=pypdl_instance) as mock_cls:
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter = BilibiliAdapter(cookie=cookie)
                    adapter.download(video_id, "https://www.bilibili.com/video/" + video_id)

        return mock_cls, pypdl_instance, out

    return _make


class TestSegmentedPath:
    def test_https_bestaudio_uses_pypdl_with_segments_and_retries(self, pypdl_setup) -> None:
        mock_cls, pypdl_instance, _ = pypdl_setup("BVseg")
        mock_cls.assert_called_once_with(allow_reuse=True)
        kwargs = pypdl_instance.start.call_args.kwargs
        assert kwargs["multisegment"] is True
        assert kwargs["segments"] == 4
        assert kwargs["retries"] == 3
        assert kwargs["block"] is True
        assert kwargs["display"] is False
        assert kwargs["url"].endswith("/audio.m4s")
        assert kwargs["file_path"].endswith(".m4a")
        pypdl_instance.shutdown.assert_called_once()

    def test_https_bestaudio_passes_cookie_and_referer_headers(self, pypdl_setup) -> None:
        _, pypdl_instance, _ = pypdl_setup("BVhdr", cookie="SESSID=abc; bili_jct=xyz")
        headers = pypdl_instance.start.call_args.kwargs["headers"]
        assert headers["Cookie"] == "SESSID=abc; bili_jct=xyz"
        assert headers["Referer"] == "https://www.bilibili.com"

    def test_https_bestaudio_referer_default_when_no_cookie(self, pypdl_setup) -> None:
        _, pypdl_instance, _ = pypdl_setup("BVref")
        headers = pypdl_instance.start.call_args.kwargs["headers"]
        assert "Cookie" not in headers
        assert headers["Referer"] == "https://www.bilibili.com"

    def test_https_bestaudio_short_file_raises_download_error(self, tmp_path) -> None:
        """If on-disk size doesn't match expected, DownloadError is raised and
        the short file is removed so a retry starts clean."""
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        (tmp_path / "downloads" / "BVshort.m4a").write_bytes(b"short")  # 5 bytes
        adapter = BilibiliAdapter(cookie="")
        info = _https_info()
        info["filesize"] = 1234

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance
            with patch("bibilab.adapters.bilibili.Pypdl", return_value=MagicMock()):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    with pytest.raises(DownloadError) as exc_info:
                        adapter.download("BVshort", "https://www.bilibili.com/video/BVshort")
        assert "content-length" in str(exc_info.value).lower()
        assert not (tmp_path / "downloads" / "BVshort.m4a").exists()

    def test_https_bestaudio_no_expected_size_skips_size_check(self, tmp_path) -> None:
        """If yt-dlp didn't surface filesize, pypdl's own validation is the
        only guarantee — download() must not raise a spurious size mismatch."""
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        out = tmp_path / "downloads" / "BVnoinfo.m4a"
        out.write_bytes(b"some bytes")

        adapter = BilibiliAdapter(cookie="")
        info = _https_info(filesize=None)
        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance
            with patch("bibilab.adapters.bilibili.Pypdl", return_value=MagicMock()):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    result = adapter.download("BVnoinfo", "https://www.bilibili.com/video/BVnoinfo")
        assert result == out
        assert out.exists()