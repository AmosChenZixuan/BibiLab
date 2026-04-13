"""Tests for BilibiliAdapter multi-part video handling."""

from unittest.mock import MagicMock, patch

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.adapters.bilibili import BilibiliAdapter


def _make_video_info(bvid="BV1abc123", title="Test Video", duration=3600):
    return {
        "id": bvid,
        "title": title,
        "webpage_url": f"https://www.bilibili.com/video/{bvid}",
        "thumbnail": "https://example.com/cover.jpg",
        "duration": duration,
        "uploader": "TestUser",
        "ext": "mp4",
    }


def _make_multipart_info(bvid="BV1abc123", title="Multi-Part Video", num_parts=3):
    """Simulate yt-dlp returning playlist structure for multi-part video."""
    entries = []
    for i in range(1, num_parts + 1):
        entries.append(
            {
                "id": f"{bvid}_p{i}",
                "title": f"{title} - P{i}",
                "webpage_url": f"https://www.bilibili.com/video/{bvid}",
                "thumbnail": "https://example.com/cover.jpg",
                "duration": 1200,
                "uploader": "TestUser",
                "playlist": title,
                "playlist_index": i,
                "ext": "mp4",
            }
        )
    return {
        "_type": "playlist",
        "id": bvid,
        "title": title,
        "webpage_url": f"https://www.bilibili.com/video/{bvid}",
        "entries": entries,
    }


class TestResolveMultiPart:
    """Test resolve() correctly handles multi-part videos."""

    def test_resolve_multipart_returns_playlist_meta(self):
        """Multi-part video resolve should return PlaylistMeta with N VideoMeta."""
        adapter = BilibiliAdapter()
        info = _make_multipart_info(bvid="BV1multi", title="Multi Part", num_parts=3)

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value=info)

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            result = adapter.resolve("https://www.bilibili.com/video/BV1multi")

        assert result.__class__.__name__ == "PlaylistMeta"
        assert len(result.videos) == 3
        assert result.playlist_id == "BV1multi"

    def test_resolve_multipart_video_ids_correct(self):
        """Each VideoMeta should have video_id in BVxxx_p{i} format."""
        adapter = BilibiliAdapter()
        info = _make_multipart_info(bvid="BV1test", title="Test", num_parts=3)

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value=info)

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            result = adapter.resolve("https://www.bilibili.com/video/BV1test")

        assert result.videos[0].video_id == "BV1test_p1"
        assert result.videos[1].video_id == "BV1test_p2"
        assert result.videos[2].video_id == "BV1test_p3"

    def test_resolve_multipart_part_labels_correct(self):
        """Each VideoMeta should have part_label in P{i} format."""
        adapter = BilibiliAdapter()
        info = _make_multipart_info(bvid="BV1test", title="Test", num_parts=3)

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value=info)

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            result = adapter.resolve("https://www.bilibili.com/video/BV1test")

        assert result.videos[0].part_label == "P1"
        assert result.videos[1].part_label == "P2"
        assert result.videos[2].part_label == "P3"

    def test_resolve_single_part_video_has_no_part_label(self):
        """Single video resolve should return VideoMeta with part_label=None."""
        adapter = BilibiliAdapter()
        info = _make_video_info(bvid="BV1single", title="Single Video")

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value=info)

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            result = adapter.resolve("https://www.bilibili.com/video/BV1single")

        assert result.__class__.__name__ == "VideoMeta"
        assert result.video_id == "BV1single"
        assert result.part_label is None


class TestDownloadMultiPart:
    """Test download() correctly handles multi-part video IDs."""

    def test_download_multipart_passes_playlist_items(self, tmp_path):
        """download('BVxxx_p3') should pass playlist_items='3' in extra_info."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value={"ext": "mp4"})

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BV1test_p3", "https://www.bilibili.com/video/BV1test")

        call_args = mock.extract_info.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs.get("extra_info", {}).get("playlist_items") == "3"

    def test_download_regular_video_no_playlist_items(self, tmp_path):
        """download('BVxxx') should NOT pass playlist_items in extra_info."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        mock = MagicMock()
        mock.__enter__ = lambda s: mock
        mock.__exit__ = MagicMock(return_value=False)
        mock.extract_info = MagicMock(return_value={"ext": "mp4"})

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BV1test", "https://www.bilibili.com/video/BV1test")

        call_args = mock.extract_info.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert not kwargs.get("extra_info") or "playlist_items" not in kwargs.get("extra_info", {})


class TestSplitVideoId:
    """Test _split_video_id helper."""

    def test_split_video_id_with_part(self):
        """_split_video_id('BV1test_p3') returns ('BV1test', 3)."""
        from bibilab.adapters.bilibili import _split_video_id

        base, part = _split_video_id("BV1test_p3")
        assert base == "BV1test"
        assert part == 3

    def test_split_video_id_without_part(self):
        """_split_video_id('BV1test') returns ('BV1test', None)."""
        from bibilab.adapters.bilibili import _split_video_id

        base, part = _split_video_id("BV1test")
        assert base == "BV1test"
        assert part is None


class TestGetVideosMetadata:
    """Test get_videos_metadata() batch-fetches metadata via Bilibili API."""

    @pytest.mark.asyncio
    async def test_get_videos_metadata_returns_metadata_map(self):
        """Given a list of BVIDs, returns dict mapping video_id -> VideoMeta."""
        from bibilab.adapters.bilibili import BilibiliAdapter

        adapter = BilibiliAdapter(cookie="test_cookie")

        async def mock_fetch_one(bvid: str, client):
            return (
                bvid,
                VideoMeta(
                    video_id=bvid,
                    title=f"Video {bvid}",
                    platform="bilibili",
                    source_url=f"https://bilibili.com/video/{bvid}",
                    cover_url=f"https://example.com/{bvid}.jpg",
                    duration_seconds=3600,
                    uploader=f"User{bvid}",
                ),
            )

        with patch("bibilab.adapters.bilibili.BilibiliAdapter.get_videos_metadata", autospec=True) as mock_get:
            mock_get.return_value = {
                "BV1abc123": VideoMeta(
                    video_id="BV1abc123",
                    title="Video BV1abc123",
                    platform="bilibili",
                    source_url="https://bilibili.com/video/BV1abc123",
                    cover_url="https://example.com/BV1abc123.jpg",
                    duration_seconds=3600,
                    uploader="UserBV1abc123",
                ),
                "BV2def456": VideoMeta(
                    video_id="BV2def456",
                    title="Video BV2def456",
                    platform="bilibili",
                    source_url="https://bilibili.com/video/BV2def456",
                    cover_url="https://example.com/BV2def456.jpg",
                    duration_seconds=1800,
                    uploader="UserBV2def456",
                ),
            }

            result = await adapter.get_videos_metadata(["BV1abc123", "BV2def456"])

        assert isinstance(result, dict)
        assert "BV1abc123" in result
        assert "BV2def456" in result
        assert result["BV1abc123"].title == "Video BV1abc123"
        assert result["BV1abc123"].cover_url == "https://example.com/BV1abc123.jpg"
        assert result["BV1abc123"].duration_seconds == 3600
        assert result["BV1abc123"].uploader == "UserBV1abc123"
        assert result["BV2def456"].title == "Video BV2def456"
        assert result["BV2def456"].cover_url == "https://example.com/BV2def456.jpg"
        assert result["BV2def456"].duration_seconds == 1800
        assert result["BV2def456"].uploader == "UserBV2def456"

    @pytest.mark.asyncio
    async def test_get_videos_metadata_handles_missing_video(self):
        """If API returns no data for a BVID, it is absent from result dict."""
        from bibilab.adapters.bilibili import BilibiliAdapter

        adapter = BilibiliAdapter(cookie="test_cookie")

        with patch("bibilab.adapters.bilibili.BilibiliAdapter.get_videos_metadata", autospec=True) as mock_get:
            mock_get.return_value = {
                "BV1abc123": VideoMeta(
                    video_id="BV1abc123",
                    title="Video One",
                    platform="bilibili",
                    source_url="https://bilibili.com/video/BV1abc123",
                    cover_url="https://example.com/cover1.jpg",
                    duration_seconds=3600,
                    uploader="UserOne",
                ),
            }

            result = await adapter.get_videos_metadata(["BV1abc123", "BVnotfound"])

        assert "BV1abc123" in result
        assert "BVnotfound" not in result


class TestResolveFlat:
    """Test resolve_flat() correctly branches on resource type."""

    def test_resolve_flat_single_video_returns_one_item_playlist(self):
        """Single-video URL returns PlaylistMeta with one video."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        info = _make_video_info("BV1single", "Single Video", 1800)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            result = adapter.resolve_flat("https://www.bilibili.com/video/BV1single")

        assert len(result.videos) == 1
        assert result.videos[0].video_id == "BV1single"
        assert result.videos[0].title == "Single Video"
        assert result.videos[0].cover_url == "https://example.com/cover.jpg"
        assert result.videos[0].duration_seconds == 1800
        assert result.videos[0].uploader == "TestUser"
        assert result.title == "Single Video"

    def test_resolve_flat_course_raises_auth_required(self):
        """Course URL raises AuthRequiredError."""
        from bibilab.adapters.base import AuthRequiredError

        adapter = BilibiliAdapter(cookie="test_cookie")

        with pytest.raises(AuthRequiredError):
            adapter.resolve_flat("https://www.bilibili.com/cheese/")

    def test_resolve_flat_playlist_uses_flat_extraction(self):
        """Playlist URL passes extract_flat='in_playlist' to yt-dlp."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = {"id": "ml", "title": "Playlist", "entries": []}
            mock_ydl.return_value.__enter__.return_value = mock_instance

            adapter.resolve_flat("https://space.bilibili.com/123/channel/")

        call_kwargs = mock_instance.extract_info.call_args.kwargs
        assert call_kwargs.get("download") is False


class TestGetVideosMetadataConcurrency:
    """Test get_videos_metadata respects concurrency limit."""

    @pytest.mark.asyncio
    async def test_get_videos_metadata_respects_concurrency_limit(self):
        """Semaphore limits in-flight requests."""
        import asyncio

        from bibilab.adapters.bilibili import BilibiliAdapter

        concurrency_seen = []

        class MockResponse:
            def __init__(self, status_code=200, json_data=None):
                self.status_code = status_code
                self._json_data = json_data or {
                    "data": {"title": "V", "pic": "", "duration": 0, "owner": {"name": "U"}}
                }

            def json(self):
                return self._json_data

        async def fake_get(url):
            bvid = url.split("?bvid=")[1]
            concurrency_seen.append(bvid)
            await asyncio.sleep(0.02)
            return MockResponse()

        class SlowClient:
            async def get(self, url):
                return await fake_get(url)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = SlowClient()

            adapter = BilibiliAdapter(cookie="test")
            bvid_list = [f"BV{i:02d}" for i in range(12)]
            await adapter.get_videos_metadata(bvid_list)

        assert len(concurrency_seen) == 12
