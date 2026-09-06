"""Tests for cleanup.purge_download_files — .part hygiene across download attempts,
plus _find_cached_video / _evict_cache_if_needed for the download-stage cache."""

from pathlib import Path

from bibilab.cleanup import _find_cached_video, purge_download_files


def test_purge_removes_target_files_only(downloads_dir: Path):
    main = downloads_dir / "BV1abc.m4a"
    part = downloads_dir / "BV1abc.m4a.part"
    other = downloads_dir / "BV1xyz.m4a"
    for f in (main, part, other):
        f.write_bytes(b"x")

    purge_download_files("BV1abc")

    assert not main.exists()
    assert not part.exists()
    assert other.exists()


# ---------------------------------------------------------------------------
# _find_cached_video — cache lookup for _stage_download
# ---------------------------------------------------------------------------


def test_find_cached_video_returns_none_on_miss(downloads_dir: Path):
    """AC1 — no file matching video_id returns None."""
    assert _find_cached_video("BVnomatch") is None


def test_find_cached_video_returns_path_on_hit(downloads_dir: Path):
    """AC2 — a non-empty, non-.part file matching video_id is returned."""
    cached = downloads_dir / "BVhit.mp4"
    cached.write_bytes(b"video bytes")

    result = _find_cached_video("BVhit")

    assert result == cached


def test_find_cached_video_skips_part_files(downloads_dir: Path):
    """AC3 — only a .part file matches → cache miss (real download must run)."""
    part = downloads_dir / "BVpart.mp4.part"
    part.write_bytes(b"partial bytes")

    assert _find_cached_video("BVpart") is None


def test_find_cached_video_rejects_zero_size_files(downloads_dir: Path):
    """AC4 — an empty file at the cache path is treated as miss (stale stub)."""
    empty = downloads_dir / "BVempty.mp4"
    empty.write_bytes(b"")

    assert _find_cached_video("BVempty") is None


def test_find_cached_video_prefers_non_part_when_both_exist(downloads_dir: Path):
    """When both a stale .part and a real file exist, the real file wins."""
    part = downloads_dir / "BVboth.mp4.part"
    part.write_bytes(b"partial")
    real = downloads_dir / "BVboth.mp4"
    real.write_bytes(b"full")

    result = _find_cached_video("BVboth")

    assert result == real
