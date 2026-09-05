"""Tests for YouTubeAdapter resolve/metadata/download behavior (yt-dlp mocked)."""

import sys
from unittest.mock import patch

import pytest
import yt_dlp

from bibilab.adapters.base import AuthRequiredError, DownloadError
from bibilab.adapters.youtube import YouTubeAdapter


def _make_run_ytdlp(captured_argv: list, *, stdout: str = "/out/video.webm", stderr: str = "", returncode: int = 0):
    """Stand-in for run_ytdlp that records the argv it was called with."""

    async def fake_run_ytdlp(argv):
        captured_argv.append(argv)
        return stdout, stderr, returncode

    return fake_run_ytdlp


def _video_info(vid="dQw4w9WgXcQ", title="Test Video", duration=212):
    return {
        "id": vid,
        "title": title,
        "webpage_url": f"https://www.youtube.com/watch?v={vid}",
        "thumbnail": "https://i.ytimg.com/vi/x/hq720.jpg",
        "duration": duration,
        "uploader": "TestChannel",
        "ext": "webm",
    }


def _playlist_info(plid="PLabc", title="Test Playlist", count=3):
    return {
        "_type": "playlist",
        "id": plid,
        "title": title,
        "webpage_url": f"https://www.youtube.com/playlist?list={plid}",
        "entries": [
            {
                "id": f"vid{i}",
                "title": f"Video {i}",
                "url": f"https://www.youtube.com/watch?v=vid{i}",
                "thumbnails": [{"url": f"https://i.ytimg.com/vi/vid{i}/hq720.jpg"}],
                "duration": 100 * i,
                "uploader": "TestChannel",
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
    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", _mock_ydl(info=_video_info())):
        result = YouTubeAdapter().resolve_flat("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert len(result.videos) == 1
    v = result.videos[0]
    assert v.video_id == "dQw4w9WgXcQ"
    assert v.title == "Test Video"
    assert v.platform == "youtube"
    assert v.duration_seconds == 212
    assert v.uploader == "TestChannel"
    assert v.part_label is None


def test_resolve_playlist():
    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", _mock_ydl(info=_playlist_info())):
        result = YouTubeAdapter().resolve_flat("https://www.youtube.com/playlist?list=PLabc")
    assert result.playlist_id == "PLabc"
    assert result.title == "Test Playlist"
    assert [v.video_id for v in result.videos] == ["vid1", "vid2", "vid3"]
    assert result.videos[0].duration_seconds == 100
    assert result.videos[0].cover_url == "https://i.ytimg.com/vi/vid1/hq720.jpg"
    assert all(v.platform == "youtube" for v in result.videos)


@pytest.mark.parametrize(
    "message",
    [
        "Sign in to confirm you're not a bot",
        "This video is private",
        "Sign in to confirm your age",
        "Join this channel to get access to members-only content",
    ],
)
def test_resolve_auth_errors(message: str):
    err = yt_dlp.utils.DownloadError(message)
    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", _mock_ydl(error=err)):
        with pytest.raises(AuthRequiredError):
            YouTubeAdapter().resolve_flat("https://www.youtube.com/watch?v=x")


def test_resolve_other_error_is_download_error():
    err = yt_dlp.utils.DownloadError("Video unavailable")
    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", _mock_ydl(error=err)):
        with pytest.raises(DownloadError):
            YouTubeAdapter().resolve_flat("https://www.youtube.com/watch?v=x")


def test_resolve_single_video_without_id_yields_no_videos():
    info = _video_info()
    del info["id"]
    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", _mock_ydl(info=info)):
        result = YouTubeAdapter().resolve_flat("https://www.youtube.com/watch?v=x")
    assert result.videos == []


@pytest.mark.asyncio
async def test_metadata_batch_survives_unexpected_exception():
    class MockYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            if "boom" in url:
                raise OSError("connection reset")
            return _video_info(vid=url.rsplit("=", 1)[1])

    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", MockYDL):
        metadata, _ = await YouTubeAdapter().get_videos_metadata(["ok1", "boom2", "ok3"])

    assert set(metadata) == {"ok1", "ok3"}


@pytest.mark.asyncio
async def test_metadata_batch_omits_failed_ids():
    calls = {}

    class MockYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            calls[url] = True
            if "bad" in url:
                raise yt_dlp.utils.DownloadError("Video unavailable")
            vid = url.rsplit("=", 1)[1]
            return _video_info(vid=vid)

    with patch("bibilab.adapters.youtube.yt_dlp.YoutubeDL", MockYDL):
        metadata, expanded = await YouTubeAdapter().get_videos_metadata(["good1", "bad2", "good3"])

    assert set(metadata) == {"good1", "good3"}
    assert metadata["good1"].title == "Test Video"
    assert expanded == {}
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_download_opts_and_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []
    expected_path = tmp_path / "downloads" / "vidX.webm"
    with (
        patch("bibilab.adapters.youtube.run_ytdlp", _make_run_ytdlp(captured, stdout=str(expected_path))),
        patch("bibilab.adapters._ytdlp_common.shutil.which", return_value=None),
    ):
        path = await YouTubeAdapter().download("vidX", "https://www.youtube.com/watch?v=vidX", connections=4)
    assert path == tmp_path / "downloads" / "vidX.webm"
    argv = captured[0]
    assert argv[argv.index("-f") + 1] == "bestaudio/best"
    assert "--downloader" not in argv


@pytest.mark.asyncio
async def test_download_uses_aria2c_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []
    with (
        patch("bibilab.adapters.youtube.run_ytdlp", _make_run_ytdlp(captured)),
        patch("bibilab.adapters._ytdlp_common.shutil.which", return_value="/usr/bin/aria2c"),
    ):
        await YouTubeAdapter().download("vidX", "https://www.youtube.com/watch?v=vidX", connections=8)
    argv = captured[0]
    assert argv[argv.index("--downloader") + 1] == "aria2c"
    assert "-x8" in argv[argv.index("--downloader-args") + 1]


@pytest.mark.asyncio
async def test_download_invokes_yt_dlp_as_a_module_of_this_interpreter(tmp_path, monkeypatch):
    """Never a bare `yt-dlp` binary — it may not be on PATH in a container or
    a uv-managed venv."""
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []

    async def fake_run_subprocess(argv, **kwargs):
        captured.append(argv)
        return "/out/video.webm", "", 0

    with patch("bibilab.adapters._ytdlp_common._run_subprocess", fake_run_subprocess):
        await YouTubeAdapter().download("vidX", "https://www.youtube.com/watch?v=vidX", connections=4)

    assert captured[0][:3] == [sys.executable, "-m", "yt_dlp"]


@pytest.mark.asyncio
async def test_download_auth_family_stderr_raises_auth_required(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []
    with patch(
        "bibilab.adapters.youtube.run_ytdlp",
        _make_run_ytdlp(captured, stderr="Sign in to confirm you're not a bot", returncode=1),
    ):
        with pytest.raises(AuthRequiredError):
            await YouTubeAdapter().download("vidX", "https://www.youtube.com/watch?v=vidX", connections=4)


@pytest.mark.asyncio
async def test_download_other_stderr_raises_download_error_with_ansi_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBILAB_HOME", str(tmp_path))
    captured: list = []
    with patch(
        "bibilab.adapters.youtube.run_ytdlp",
        _make_run_ytdlp(captured, stderr="\x1b[31mERROR: video unavailable\x1b[0m", returncode=1),
    ):
        with pytest.raises(DownloadError) as exc_info:
            await YouTubeAdapter().download("vidX", "https://www.youtube.com/watch?v=vidX", connections=4)
    assert exc_info.value.message == "ERROR: video unavailable"
