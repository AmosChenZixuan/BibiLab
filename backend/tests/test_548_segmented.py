"""Tests for #548: segmented pypdl path is used for https bestaudio.

AC1 — segmented pypdl path is taken when info has protocol=https + no fragments.
AC7 — cookie + Referer headers reach pypdl so direct-CDN requests stay authed.
"""

from unittest.mock import MagicMock, patch

import pytest

from bibilab.adapters.bilibili import BilibiliAdapter


def _https_info(url: str = "https://example.com/audio.m4s", headers: dict | None = None) -> dict:
    return {
        "url": url,
        "ext": "m4a",
        "protocol": "https",
        "fragments": None,
        "http_headers": headers or {"User-Agent": "yt-dlp/test"},
        "filesize": 1234,
    }


class TestSegmentedPath:
    """AC1 + AC7: when yt-dlp returns a direct https bestaudio URL, pypdl runs."""

    def test_https_bestaudio_uses_pypdl_with_segments_and_retries(self, tmp_path):
        """Direct https URL goes to pypdl.start with multisegment=True,
        segments=4 (default), retries=3, block=True."""
        # Pre-create the expected output so the post-check passes (mocks pypdl
        # without actually writing the file).
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        (tmp_path / "downloads" / "BVseg.m4a").write_bytes(b"x" * 1234)

        adapter = BilibiliAdapter(cookie="")
        fake_pypdl_instance = MagicMock()

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = _https_info()
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=fake_pypdl_instance) as mock_cls:
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter.download("BVseg", "https://www.bilibili.com/video/BVseg")

        mock_cls.assert_called_once_with(allow_reuse=True)
        fake_pypdl_instance.start.assert_called_once()
        kwargs = fake_pypdl_instance.start.call_args.kwargs
        assert kwargs["multisegment"] is True
        assert kwargs["segments"] == 4  # config default
        assert kwargs["retries"] == 3
        assert kwargs["block"] is True
        assert kwargs["display"] is False
        assert kwargs["url"].endswith("/audio.m4s")
        assert kwargs["file_path"].endswith(".m4a")
        fake_pypdl_instance.shutdown.assert_called_once()

    def test_https_bestaudio_passes_cookie_and_referer_headers(self, tmp_path):
        """AC7: cookie config + bilibili Referer must reach pypdl headers."""
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        (tmp_path / "downloads" / "BVhdr.m4a").write_bytes(b"x" * 1234)

        adapter = BilibiliAdapter(cookie="SESSID=abc; bili_jct=xyz")
        fake_pypdl_instance = MagicMock()

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = _https_info()
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=fake_pypdl_instance):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter.download("BVhdr", "https://www.bilibili.com/video/BVhdr")

        headers = fake_pypdl_instance.start.call_args.kwargs["headers"]
        assert headers["Cookie"] == "SESSID=abc; bili_jct=xyz"
        assert headers["Referer"] == "https://www.bilibili.com"

    def test_https_bestaudio_referer_default_when_no_cookie(self, tmp_path):
        """Without a cookie, the Referer still goes to bilibili.com so direct
        CDN requests are not rejected as hotlinks."""
        (tmp_path / "downloads").mkdir(parents=True, exist_ok=True)
        (tmp_path / "downloads" / "BVref.m4a").write_bytes(b"x" * 1234)

        adapter = BilibiliAdapter(cookie="")
        fake_pypdl_instance = MagicMock()

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = _https_info()
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=fake_pypdl_instance):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    adapter.download("BVref", "https://www.bilibili.com/video/BVref")

        headers = fake_pypdl_instance.start.call_args.kwargs["headers"]
        assert "Cookie" not in headers
        assert headers["Referer"] == "https://www.bilibili.com"

    def test_https_bestaudio_short_file_raises_download_error(self, tmp_path):
        """AC3 (mocked): if on-disk size doesn't match expected, DownloadError
        is raised and the short file is removed."""
        from bibilab.adapters.base import DownloadError

        # Pre-create a file that simulates a partial pypdl result.
        out = tmp_path / "downloads" / "BVshort.m4a"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"short")  # 5 bytes

        adapter = BilibiliAdapter(cookie="")
        info = _https_info()
        info["filesize"] = 1234  # expected size larger than actual

        fake_pypdl_instance = MagicMock()
        # Simulate pypdl completing but leaving a short file on disk.

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=fake_pypdl_instance):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    with pytest.raises(DownloadError) as exc_info:
                        adapter.download("BVshort", "https://www.bilibili.com/video/BVshort")

        assert "content-length" in str(exc_info.value).lower()
        # Short file removed so a retry starts clean.
        assert not out.exists() or out.stat().st_size == 1234

    def test_https_bestaudio_no_expected_size_skips_size_check(self, tmp_path):
        """If yt-dlp didn't surface filesize, pypdl's own validation is the
        only guarantee. download() must not raise a spurious size mismatch."""
        adapter = BilibiliAdapter(cookie="")
        info = _https_info()
        info.pop("filesize", None)

        out = tmp_path / "downloads" / "BVnoinfo.m4a"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"some bytes")

        fake_pypdl_instance = MagicMock()
        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with patch("bibilab.adapters.bilibili.Pypdl", return_value=fake_pypdl_instance):
                with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                    result = adapter.download("BVnoinfo", "https://www.bilibili.com/video/BVnoinfo")

        assert result == out
        assert out.exists()