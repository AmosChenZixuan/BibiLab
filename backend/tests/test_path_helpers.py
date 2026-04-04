from pathlib import Path


def test_resolve_storage_path_relative(tmp_bibilab_home: Path):
    from bibilab.config import bibilab_home, resolve_storage_path

    home = bibilab_home()
    result = resolve_storage_path("notes/BV1abc.md")
    assert result == home / "notes" / "BV1abc.md"
