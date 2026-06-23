"""Tests for yt-dlp DownloadError → AuthRequiredError mapping on resolve path.

The download() resolve call surfaces a 403 / login / 412 as AuthRequiredError
so the UI can route to the QR login flow — not a generic DownloadError that
masks the cause.
"""

from unittest.mock import patch

import pytest
import yt_dlp

from bibilab.adapters.base import AuthRequiredError, DownloadError
from bibilab.adapters.bilibili import BilibiliAdapter


class _ResolveOnlyYDL:
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
        "network timeout",
    ],
)
def test_resolve_error_mapping(tmp_path, msg: str) -> None:
    adapter = BilibiliAdapter(cookie="")
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", lambda opts: _ResolveOnlyYDL(msg)):
        with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
            if _AUTH_KEYWORDS := ("403", "log in", "login", "412"):
                auth_like = any(k in msg.lower() for k in _AUTH_KEYWORDS)
            else:
                auth_like = False
            if auth_like:
                with pytest.raises(AuthRequiredError) as exc_info:
                    adapter.download("BVx", "https://www.bilibili.com/video/BVx")
                assert exc_info.value.resource_type == "video"
            else:
                with pytest.raises(DownloadError):
                    adapter.download("BVx", "https://www.bilibili.com/video/BVx")