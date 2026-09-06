"""Tests for cleanup.purge_download_files — .part hygiene across download attempts,
plus _find_cached_video / _evict_cache_if_needed for the download-stage cache."""

import os
from pathlib import Path

from bibilab.cleanup import _evict_cache_if_needed, _find_cached_video, purge_download_files


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


# ---------------------------------------------------------------------------
# _evict_cache_if_needed — LRU by mtime, 10 GB cap
# ---------------------------------------------------------------------------


def _set_mtime(path: Path, seconds: float) -> None:
    """Set mtime/atime explicitly so LRU ordering is deterministic."""
    os.utime(path, (seconds, seconds))


def test_evict_noop_when_under_cap(downloads_dir: Path, monkeypatch):
    """Cache under cap → nothing is deleted."""
    monkeypatch.setattr("bibilab.cleanup.CACHE_MAX_BYTES", 10_000)
    files = []
    for i in range(3):
        f = downloads_dir / f"BVold{i}.mp4"
        f.write_bytes(b"x" * 100)
        _set_mtime(f, 1_000_000 + i)
        files.append(f)

    _evict_cache_if_needed()

    for f in files:
        assert f.exists()


def test_evict_drops_oldest_first(downloads_dir: Path, monkeypatch):
    """AC7 + AC8 — when over cap, oldest by mtime is deleted first; newest survives."""
    # 3 files of 100 bytes each → 300 bytes total. Cap = 200 → evict 100 bytes → drop 1 oldest.
    monkeypatch.setattr("bibilab.cleanup.CACHE_MAX_BYTES", 200)
    oldest = downloads_dir / "BVoldest.mp4"
    middle = downloads_dir / "BVmiddle.mp4"
    newest = downloads_dir / "BVnewest.mp4"
    for f, ts in [(oldest, 1_000_000), (middle, 2_000_000), (newest, 3_000_000)]:
        f.write_bytes(b"x" * 100)
        _set_mtime(f, ts)

    _evict_cache_if_needed()

    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_evict_skips_part_files(downloads_dir: Path, monkeypatch):
    """AC9 — .part files are never counted toward cap or evicted."""
    monkeypatch.setattr("bibilab.cleanup.CACHE_MAX_BYTES", 100)
    real = downloads_dir / "BVreal.mp4"
    real.write_bytes(b"x" * 100)
    _set_mtime(real, 1_000_000)
    part = downloads_dir / "BVinflight.mp4.part"
    part.write_bytes(b"x" * 1_000_000)  # 1 MB — way over cap if counted
    _set_mtime(part, 999_999)  # older than real, but .part must be ignored

    _evict_cache_if_needed()

    assert real.exists(), "real file should not be evicted (under cap without .part)"
    assert part.exists(), ".part files must never be evicted — an in-flight download depends on them"


def test_evict_keeps_oversize_files(downloads_dir: Path, monkeypatch):
    """A single file larger than the cap is left alone — deleting it would
    throw away a paid-for download that nothing else can evict."""
    monkeypatch.setattr("bibilab.cleanup.CACHE_MAX_BYTES", 100)
    huge = downloads_dir / "BVhuge.mp4"
    huge.write_bytes(b"x" * 1_000)  # 10× cap
    _set_mtime(huge, 1_000_000)
    small_old = downloads_dir / "BVsmall.mp4"
    small_old.write_bytes(b"x" * 50)
    _set_mtime(small_old, 999_999)

    _evict_cache_if_needed()

    # total under cap once huge is excluded → no eviction
    assert huge.exists()
    assert small_old.exists()
