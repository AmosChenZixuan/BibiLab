"""Tests for TikTokAdapter resolve/metadata/download behavior (yt-dlp mocked)."""

from unittest.mock import patch

import pytest
import yt_dlp

from bibilab.adapters.base import AuthRequiredError, DownloadError
from bibilab.adapters.tiktok import TikTokAdapter, _truncate_caption


def _video_info(vid="7371330159376370462", title="Short caption", duration=21):
    return {
        "id": vid,
        "title": title,
        "webpage_url": f"https://www.tiktok.com/@someuser/video/{vid}",
        "thumbnail": "https://p16-sign-va.tiktokcdn.com/obj/thumb.jpg",
        "duration": duration,
        "uploader": "someuser",
        "ext": "mp4",
    }


def _collection_info(cid="7111887189571160875", count=3):
    return {
        "_type": "playlist",
        "id": cid,
        "title": "user-collection",
        "webpage_url": f"https://www.tiktok.com/@someuser/collection/x-{cid}",
        "entries": [
            {
                "id": f"71{i}",
                "title": f"Clip {i}",
                "url": f"https://www.tiktok.com/@someuser/video/71{i}",
                "thumbnails": [
                    {"url": "https://p16.tiktokcdn.com/small.jpg", "width": 100, "height": 100},
                    {"url": "https://p16.tiktokcdn.com/big.jpg", "width": 720, "height": 1280},
                ],
                "duration": 10 * i,
                "uploader": "someuser",
            }
            for i in range(1, count + 1)
        ],
    }


def _mock_ydl(info=None, error=None, captured_opts=None):
    class MockYDL:
        def __init__(self, opts):
            if captured_opts is not None:
                captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            if error is not None:
                raise error
            return info

    return MockYDL


def test_resolve_single_video():
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(info=_video_info())):
        result = TikTokAdapter().resolve_flat("https://vm.tiktok.com/ZMabcdef/")
    assert len(result.videos) == 1
    v = result.videos[0]
    assert v.platform == "tiktok"
    assert v.video_id == "7371330159376370462"
    assert v.duration_seconds == 21
    assert v.part_label is None


def test_resolve_collection():
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(info=_collection_info())):
        result = TikTokAdapter().resolve_flat("https://www.tiktok.com/@someuser/collection/x-7111887189571160875")
    assert result.playlist_id == "7111887189571160875"
    assert [v.video_id for v in result.videos] == ["711", "712", "713"]
    # largest-area thumbnail picked, not list order
    assert result.videos[0].cover_url == "https://p16.tiktokcdn.com/big.jpg"
    assert all(v.platform == "tiktok" for v in result.videos)


def test_caption_truncated_on_word_boundary():
    long_caption = ("word " * 40).strip()  # 199 chars
    info = _video_info(title=long_caption)
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(info=info)):
        result = TikTokAdapter().resolve_flat("https://www.tiktok.com/@u/video/1")
    title = result.videos[0].title
    assert len(title) <= 121  # 120 + ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")


def test_caption_without_spaces_hard_cut():
    info = _video_info(title="字" * 200)
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(info=info)):
        result = TikTokAdapter().resolve_flat("https://www.tiktok.com/@u/video/1")
    title = result.videos[0].title
    assert len(title) == 121
    assert title.endswith("…")


def test_short_caption_untouched():
    assert _truncate_caption("hello world") == "hello world"


def test_login_wall_maps_to_auth_required():
    err = yt_dlp.utils.DownloadError("TikTok is requiring login for access to this content")
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(error=err)):
        with pytest.raises(AuthRequiredError):
            TikTokAdapter().resolve_flat("https://www.tiktok.com/@u/video/1")


def test_image_post_maps_to_specific_error():
    err = yt_dlp.utils.DownloadError("ERROR: No video formats found!")
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(error=err)):
        with pytest.raises(DownloadError) as exc_info:
            TikTokAdapter().resolve_flat("https://www.tiktok.com/@u/photo/1")
    assert "image post" in str(exc_info.value)


def test_generic_error_carries_upgrade_hint():
    err = yt_dlp.utils.DownloadError("Unable to extract webpage video data")
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(error=err)):
        with pytest.raises(DownloadError) as exc_info:
            TikTokAdapter().resolve_flat("https://www.tiktok.com/@u/video/1")
    assert "yt-dlp" in str(exc_info.value)


@pytest.mark.asyncio
async def test_metadata_batch_omits_failed_and_survives_unexpected():
    class MockYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            if "111" in url:
                raise yt_dlp.utils.DownloadError("Unable to extract")
            if "222" in url:
                raise OSError("connection reset")
            return _video_info(vid=url.rsplit("/", 1)[1])

    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", MockYDL):
        metadata, expanded = await TikTokAdapter().get_videos_metadata(["111", "222", "333"])

    assert set(metadata) == {"333"}
    assert expanded == {}


def test_download_path_and_no_aria2c(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []
    with patch("bibilab.adapters.tiktok.yt_dlp.YoutubeDL", _mock_ydl(info={"ext": "mp4"}, captured_opts=captured)):
        path = TikTokAdapter().download("71x", "https://www.tiktok.com/@u/video/71x", connections=8)
    assert path == tmp_path / "downloads" / "71x.mp4"
    opts = captured[0]
    assert opts["format"] == "bestaudio/best"
    # small files — native downloader, no aria2c branch
    assert "external_downloader" not in opts


def test_pick_thumbnail_dimensionless_falls_back_to_last():
    from bibilab.adapters._ytdlp_common import pick_thumbnail

    entry = {"thumbnails": [{"url": "https://cdn/first.jpg"}, {"url": "https://cdn/last.jpg"}]}
    assert pick_thumbnail(entry) == "https://cdn/last.jpg"
