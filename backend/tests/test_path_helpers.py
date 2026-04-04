from pathlib import Path

import pytest


def test_resolve_storage_path_relative(tmp_bibilab_home: Path):
    from bibilab.config import resolve_storage_path

    result = resolve_storage_path("notes/BV1abc.md")
    assert result == tmp_bibilab_home / "notes" / "BV1abc.md"


def test_resolve_storage_path_absolute_legacy():
    from bibilab.config import resolve_storage_path

    result = resolve_storage_path("/tmp/old/path.md")
    assert result == Path("/tmp/old/path.md")


def test_relative_to_bibilab_home():
    from bibilab.config import bibilab_home, relative_to_bibilab_home

    home = bibilab_home()
    result = relative_to_bibilab_home(home / "notes" / "BV1abc.md")
    assert result == "notes/BV1abc.md"


def test_relative_to_bibilab_home_raises_on_outside_path(tmp_bibilab_home: Path):
    from bibilab.config import relative_to_bibilab_home

    with pytest.raises(ValueError):
        relative_to_bibilab_home(Path("/tmp/some_file.md"))
