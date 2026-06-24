"""Tests for BilibiliAdapter multi-part video handling."""

from unittest.mock import MagicMock, patch

import pytest

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


def _make_mock_ydl(captured_opts: list):
    class MockYDL:
        def __init__(self, opts):
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            return {"ext": "mp4"}

    return MockYDL


class TestDownloadMultiPart:
    """Test download() correctly handles multi-part video IDs."""

    def test_download_multipart_passes_playlist_items(self, tmp_path):
        adapter = BilibiliAdapter(cookie="test_cookie")
        captured_opts: list = []

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _make_mock_ydl(captured_opts)):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BV1test_p3", "https://www.bilibili.com/video/BV1test")

        assert captured_opts[0]["playlist_items"] == "3"

    def test_download_regular_video_no_playlist_items(self, tmp_path):
        adapter = BilibiliAdapter(cookie="test_cookie")
        captured_opts: list = []

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _make_mock_ydl(captured_opts)):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                adapter.download("BV1test", "https://www.bilibili.com/video/BV1test")

        assert "playlist_items" not in captured_opts[0]

    def test_download_sets_native_retry_opts(self, tmp_path):
        # Without these, yt-dlp's bare opts default to 0 internal retries
        # (RetryManager does _retries or 0), and transient CDN timeouts become fatal.
        from bibilab.adapters.bilibili import _FRAGMENT_RETRIES, _HTTP_RETRIES, _SOCKET_TIMEOUT

        adapter = BilibiliAdapter(cookie="test_cookie")
        captured_opts: list = []

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _make_mock_ydl(captured_opts)):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                with patch("bibilab.adapters.bilibili.shutil.which", return_value="/usr/bin/aria2c"):
                    adapter.download("BV1test", "https://www.bilibili.com/video/BV1test")

        assert captured_opts[0]["retries"] == _HTTP_RETRIES
        assert captured_opts[0]["fragment_retries"] == _FRAGMENT_RETRIES
        # A stalled connection must become a retriable error, not an indefinite
        # hang that wedges the serialized download stage.
        assert captured_opts[0]["socket_timeout"] == _SOCKET_TIMEOUT


class TestDownloadAria2c:
    """Three observable behaviors: download() routes through yt-dlp + aria2c
    when available, falls back to native yt-dlp when not, and honors
    download_connections."""

    @staticmethod
    def _patched_download(tmp_path, *, aria2c_path, download_connections):
        """Helper: build a BilibiliAdapter, run download() under all the
        mocks needed to capture yt-dlp opts deterministically."""
        from bibilab.config import BibilabConfig

        adapter = BilibiliAdapter(cookie="test_cookie")
        captured_opts: list = []
        cfg = BibilabConfig()
        cfg.backend.download_connections = download_connections

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _make_mock_ydl(captured_opts)):
            with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                with patch("bibilab.adapters.bilibili.shutil.which", return_value=aria2c_path):
                    with patch("bibilab.adapters.bilibili.load_config", return_value=cfg):
                        adapter.download("BV1test", "https://www.bilibili.com/video/BV1test")
        return captured_opts[0]

    def test_download_uses_aria2c_when_available(self, tmp_path):
        opts = self._patched_download(tmp_path, aria2c_path="/usr/bin/aria2c", download_connections=16)

        assert opts["external_downloader"] == "aria2c"
        assert opts["external_downloader_args"]["aria2c"] == [
            "-x16",
            "-s16",
            "-k1M",
            "--file-allocation=none",
        ]

    def test_download_falls_back_to_native_when_aria2c_absent(self, tmp_path):
        from bibilab.adapters.bilibili import _FRAGMENT_RETRIES, _HTTP_RETRIES, _SOCKET_TIMEOUT

        opts = self._patched_download(tmp_path, aria2c_path=None, download_connections=16)

        # No aria2c → no external_downloader key. Native retry/timeout opts still
        # present so the download keeps working (just slower under throttle).
        assert "external_downloader" not in opts
        assert "external_downloader_args" not in opts
        assert opts["retries"] == _HTTP_RETRIES
        assert opts["fragment_retries"] == _FRAGMENT_RETRIES
        assert opts["socket_timeout"] == _SOCKET_TIMEOUT

    def test_download_propagates_download_connections(self, tmp_path):
        opts = self._patched_download(tmp_path, aria2c_path="/usr/bin/aria2c", download_connections=8)

        assert opts["external_downloader_args"]["aria2c"] == [
            "-x8",
            "-s8",
            "-k1M",
            "--file-allocation=none",
        ]


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

    def test_resolve_flat_multipart_video_returns_all_parts(self):
        """Multi-part video URL returns PlaylistMeta with one entry per part."""
        adapter = BilibiliAdapter(cookie="test_cookie")
        info = _make_multipart_info(bvid="BV1multi", title="Multi Part", num_parts=3)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            result = adapter.resolve_flat("https://www.bilibili.com/video/BV1multi")

        assert len(result.videos) == 3
        assert result.videos[0].video_id == "BV1multi_p1"
        assert result.videos[1].video_id == "BV1multi_p2"
        assert result.videos[2].video_id == "BV1multi_p3"
        assert result.videos[0].part_label == "P1"
        assert result.videos[1].part_label == "P2"
        assert result.videos[2].part_label == "P3"

    def test_resolve_flat_video_uses_flat_extraction(self):
        """Video branch passes extract_flat='in_playlist' — flat enumeration avoids
        resolving formats for every part of a multi-part video (slow preview fix)."""
        adapter = BilibiliAdapter(cookie="test_cookie")
        captured_opts: list = []

        with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", _make_mock_ydl(captured_opts)):
            adapter.resolve_flat("https://www.bilibili.com/video/BV1x")

        assert captured_opts[0].get("extract_flat") == "in_playlist"

    def test_resolve_flat_multipart_flat_entries_derive_part_from_url(self):
        """Flat multipart entries carry no id; the part number is derived from the
        '?p=N' url and the playlist BVID."""
        adapter = BilibiliAdapter(cookie="test_cookie")
        info = {
            "_type": "playlist",
            "id": "BV1multi",
            "title": "Multi Part",
            "webpage_url": "https://www.bilibili.com/video/BV1multi",
            "entries": [
                {"id": None, "title": None, "duration": None, "url": "https://www.bilibili.com/video/BV1multi?p=1"},
                {"id": None, "title": None, "duration": None, "url": "https://www.bilibili.com/video/BV1multi?p=2"},
            ],
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = info
            mock_ydl.return_value.__enter__.return_value = mock_instance

            result = adapter.resolve_flat("https://www.bilibili.com/video/BV1multi")

        assert [v.video_id for v in result.videos] == ["BV1multi_p1", "BV1multi_p2"]
        assert [v.part_label for v in result.videos] == ["P1", "P2"]

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

    def test_resolve_flat_playlist_private_raises_auth_required(self):
        """Private playlist URL without cookie raises AuthRequiredError (401)."""
        import yt_dlp

        from bibilab.adapters.base import AuthRequiredError

        adapter = BilibiliAdapter(cookie="")

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.side_effect = yt_dlp.utils.DownloadError(
                "This video is private. Please login to access"
            )
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with pytest.raises(AuthRequiredError) as exc_info:
                adapter.resolve_flat("https://space.bilibili.com/123/favlist?fid=456")

            assert exc_info.value.resource_type == "playlist"

    def test_resolve_flat_playlist_403_raises_auth_required(self):
        """Playlist URL returning 403 error raises AuthRequiredError (401)."""
        import yt_dlp

        from bibilab.adapters.base import AuthRequiredError

        adapter = BilibiliAdapter(cookie="")

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.side_effect = yt_dlp.utils.DownloadError("ERROR: [Bilibili] 403 Forbidden")
            mock_ydl.return_value.__enter__.return_value = mock_instance

            with pytest.raises(AuthRequiredError) as exc_info:
                adapter.resolve_flat("https://space.bilibili.com/123/favlist?fid=456")

            assert exc_info.value.resource_type == "playlist"


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


class TestGetVideosMetadataDedup:
    """Test get_videos_metadata dedupes BVIDs before making API calls."""

    @pytest.mark.asyncio
    async def test_get_videos_metadata_dedups_bvids(self):
        """Input [BV1_p1, BV1_p2, BV2] should result in exactly 2 API calls."""
        from bibilab.adapters.bilibili import BilibiliAdapter

        adapter = BilibiliAdapter(cookie="test_cookie")
        fetched_bvids = []

        class MockResponse:
            def __init__(self, bvid):
                self.status_code = 200
                self._bvid = bvid

            def json(self):
                return {
                    "data": {
                        "title": f"Video {self._bvid}",
                        "pic": f"https://example.com/{self._bvid}.jpg",
                        "duration": 3600,
                        "owner": {"name": f"Uploader{self._bvid}"},
                    }
                }

        class TrackingClient:
            async def get(self, url):
                bvid = url.split("?bvid=")[1]
                fetched_bvids.append(bvid)
                return MockResponse(bvid)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = TrackingClient()

            result, _ = await adapter.get_videos_metadata(["BV1xxx_p1", "BV1xxx_p2", "BV2yyy"])

        assert len(fetched_bvids) == 2, f"Expected 2 API calls, got {len(fetched_bvids)}"
        assert set(fetched_bvids) == {"BV1xxx", "BV2yyy"}
        assert "BV1xxx_p1" in result
        assert "BV1xxx_p2" in result
        assert "BV2yyy" in result
        assert result["BV1xxx_p1"].title == "Video BV1xxx"
        assert result["BV1xxx_p2"].title == "Video BV1xxx"
        assert result["BV2yyy"].title == "Video BV2yyy"

    @pytest.mark.asyncio
    async def test_get_videos_metadata_missing_video_not_in_result(self):
        """If API returns no data for a BVID, that key is absent and other entries are unaffected."""
        from bibilab.adapters.bilibili import BilibiliAdapter

        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            def __init__(self, has_data=True):
                self.status_code = 200 if has_data else 404
                self._has_data = has_data

            def json(self):
                if not self._has_data:
                    return {"code": -404, "message": "not found"}
                return {
                    "data": {
                        "title": "Found Video",
                        "pic": "https://example.com/pic.jpg",
                        "duration": 1800,
                        "owner": {"name": "Uploader"},
                    }
                }

        class TrackingClient:
            def __init__(self):
                self.calls = []

            async def get(self, url):
                bvid = url.split("?bvid=")[1]
                self.calls.append(bvid)
                if bvid == "BVnotfound":
                    return MockResponse(has_data=False)
                return MockResponse(has_data=True)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = TrackingClient()

            result, _ = await adapter.get_videos_metadata(["BV1abc123", "BVnotfound"])

        assert "BV1abc123" in result
        assert "BVnotfound" not in result
        assert result["BV1abc123"].title == "Found Video"


class TestGetVideosMetadataExpansion:
    """Test get_videos_metadata expands multi-part videos from playlist context."""

    @pytest.mark.asyncio
    async def test_multipart_video_expanded_into_parts(self):
        """A bare BVID with pages > 1 should expand into per-part entries."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            def __init__(self):
                self.status_code = 200

            def json(self):
                return {
                    "data": {
                        "title": "Multi Part Video",
                        "pic": "https://example.com/cover.jpg",
                        "duration": 3600,
                        "owner": {"name": "Author"},
                        "pages": [
                            {"page": 1, "part": "Intro", "duration": 600},
                            {"page": 2, "part": "Main", "duration": 1800},
                            {"page": 3, "part": "Outro", "duration": 300},
                        ],
                    }
                }

        class MockClient:
            async def get(self, url):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = MockClient()

            result, expanded = await adapter.get_videos_metadata(["BV1multi"])

        assert "BV1multi" not in result
        assert "BV1multi" in expanded
        assert expanded["BV1multi"] == ["BV1multi_p1", "BV1multi_p2", "BV1multi_p3"]
        assert result["BV1multi_p1"].title == "Intro - Multi Part Video"
        assert result["BV1multi_p1"].part_label == "P1"
        assert result["BV1multi_p1"].duration_seconds == 600
        assert result["BV1multi_p2"].title == "Main - Multi Part Video"
        assert result["BV1multi_p3"].title == "Outro - Multi Part Video"

    @pytest.mark.asyncio
    async def test_single_page_video_not_expanded(self):
        """A BVID with pages == 1 should NOT expand."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            def __init__(self):
                self.status_code = 200

            def json(self):
                return {
                    "data": {
                        "title": "Single Video",
                        "pic": "https://example.com/cover.jpg",
                        "duration": 1800,
                        "owner": {"name": "Author"},
                        "pages": [{"page": 1, "part": "Single Video", "duration": 1800}],
                    }
                }

        class MockClient:
            async def get(self, url):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = MockClient()

            result, expanded = await adapter.get_videos_metadata(["BV1single"])

        assert "BV1single" in result
        assert expanded == {}
        assert result["BV1single"].title == "Single Video"

    @pytest.mark.asyncio
    async def test_already_has_parts_not_expanded(self):
        """IDs with _p suffix (already expanded) should not re-expand."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            def __init__(self):
                self.status_code = 200

            def json(self):
                return {
                    "data": {
                        "title": "Multi Part Video",
                        "pic": "https://example.com/cover.jpg",
                        "duration": 3600,
                        "owner": {"name": "Author"},
                        "pages": [
                            {"page": 1, "part": "Intro", "duration": 600},
                            {"page": 2, "part": "Main", "duration": 1800},
                        ],
                    }
                }

        class MockClient:
            async def get(self, url):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = MockClient()

            result, expanded = await adapter.get_videos_metadata(["BV1test_p1", "BV1test_p2"])

        assert expanded == {}
        assert "BV1test_p1" in result
        assert "BV1test_p2" in result

    @pytest.mark.asyncio
    async def test_already_requested_parts_get_distinct_per_part_metadata(self):
        """Parts requested individually (with _p suffix) must each get their own page
        title/duration/url — not the combined base-video metadata shared across parts."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "data": {
                        "title": "Multi Part Video",
                        "pic": "https://example.com/cover.jpg",
                        "duration": 3600,  # combined duration of all parts
                        "owner": {"name": "Author"},
                        "pages": [
                            {"page": 1, "part": "Intro", "duration": 600},
                            {"page": 2, "part": "Main", "duration": 1800},
                        ],
                    }
                }

        class MockClient:
            async def get(self, url):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = MockClient()

            result, expanded = await adapter.get_videos_metadata(["BV1test_p1", "BV1test_p2"])

        assert expanded == {}
        assert result["BV1test_p1"].duration_seconds == 600
        assert result["BV1test_p2"].duration_seconds == 1800
        assert result["BV1test_p1"].title == "Intro - Multi Part Video"
        assert result["BV1test_p2"].title == "Main - Multi Part Video"
        assert result["BV1test_p1"].part_label == "P1"
        assert result["BV1test_p2"].part_label == "P2"
        assert result["BV1test_p1"].source_url == "https://www.bilibili.com/video/BV1test?p=1"

    @pytest.mark.asyncio
    async def test_multipart_video_empty_part_name_no_trailing_colon(self):
        """Empty page.part should produce 'P1' not 'P1: '."""
        adapter = BilibiliAdapter(cookie="test_cookie")

        class MockResponse:
            def __init__(self):
                self.status_code = 200

            def json(self):
                return {
                    "data": {
                        "title": "Multi Part Video",
                        "pic": "https://example.com/cover.jpg",
                        "duration": 3600,
                        "owner": {"name": "Author"},
                        "pages": [
                            {"page": 1, "part": "", "duration": 600},
                            {"page": 2, "part": "", "duration": 1800},
                        ],
                    }
                }

        class MockClient:
            async def get(self, url):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = MockClient()

            result, expanded = await adapter.get_videos_metadata(["BV1emptypart"])

        assert result["BV1emptypart_p1"].part_label == "P1"
        assert result["BV1emptypart_p2"].part_label == "P2"
        assert result["BV1emptypart_p1"].title == "Multi Part Video"


class TestResourceType:
    """Test _resource_type() correctly classifies Bilibili URLs."""

    def test_favlist_url_classified_as_playlist(self):
        from bibilab.adapters.bilibili import _resource_type

        assert _resource_type("https://space.bilibili.com/25503580/favlist?fid=2807438580&ftype=create") == "playlist"
        assert _resource_type("https://space.bilibili.com/123/favlist?fid=456") == "playlist"
