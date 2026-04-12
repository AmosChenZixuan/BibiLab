"""Tests for BilibiliAdapter multi-part video handling."""

from unittest.mock import MagicMock, patch

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
