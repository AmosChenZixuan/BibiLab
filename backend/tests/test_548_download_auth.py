"""Tests for #548: auth-error mapping preserved on the resolve path.

The refactored download() calls extract_info(download=False) first. If that
resolve call surfaces a 403 / login / 412, the user must see AuthRequiredError
(needs_auth in the UI) — not a generic DownloadError that masks the cause.
"""

from unittest.mock import patch

import pytest
import yt_dlp

from bibilab.adapters.base import AuthRequiredError
from bibilab.adapters.bilibili import BilibiliAdapter


class _ResolveOnlyYDL:
    """Mock that fails on resolve with a given message; never reaches native download."""

    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        raise yt_dlp.utils.DownloadError(self.msg)


@pytest.mark.parametrize(
    "msg",
    [
        "ERROR: [Bilibili] 403 Forbidden",
        "Please log in to access this video",
        "HTTP Error 412: Precondition Failed",
    ],
)
def test_resolve_auth_error_raises_auth_required(tmp_path, msg: str) -> None:
    adapter = BilibiliAdapter(cookie="")
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", lambda opts: _ResolveOnlyYDL(msg)):
        with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
            with pytest.raises(AuthRequiredError) as exc_info:
                adapter.download("BVauth", "https://www.bilibili.com/video/BVauth")
    assert exc_info.value.resource_type == "video"


def test_resolve_non_auth_error_raises_download_error(tmp_path) -> None:
    """Non-auth DownloadError on resolve path: surface as generic DownloadError."""
    from bibilab.adapters.base import DownloadError

    adapter = BilibiliAdapter(cookie="")
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", lambda opts: _ResolveOnlyYDL("network timeout")):
        with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
            with pytest.raises(DownloadError) as exc_info:
                adapter.download("BVnet", "https://www.bilibili.com/video/BVnet")
    assert "network timeout" in str(exc_info.value).lower()