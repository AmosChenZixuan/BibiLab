"""Tests for cleanup.purge_download_files — .part hygiene across download attempts."""

from pathlib import Path

from bibilab.cleanup import purge_download_files


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
