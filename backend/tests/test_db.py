from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            yield tmp_path


@pytest.mark.asyncio
async def test_lists_table_exists(tmp_locus_home: Path):
    from locus.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lists'"
        ) as cur:
            assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_sources_table_exists(tmp_locus_home: Path):
    from locus.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
        ) as cur:
            assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_create_and_get_list(tmp_locus_home: Path):
    from locus.db import bootstrap_db, create_list, get_all_lists, get_list

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    row = await get_list("list-1")
    assert row is not None
    assert row["name"] == "ML Course"
    all_lists = await get_all_lists()
    assert len(all_lists) == 1


@pytest.mark.asyncio
async def test_write_and_get_source(tmp_locus_home: Path):
    from locus.db import (
        bootstrap_db,
        create_list,
        get_source,
        video_is_processed,
        write_source,
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="Intro to ML",
        summary="A great intro.",
        note_path="/home/user/.locus/notes/BV1abc.md",
        transcript_path="/home/user/.locus/transcripts/BV1abc.txt",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    source = await get_source("BV1abc")
    assert source is not None
    assert source["title"] == "Intro to ML"
    assert await video_is_processed("BV1abc") is True
    assert await video_is_processed("BV1xyz") is False


@pytest.mark.asyncio
async def test_delete_source(tmp_locus_home: Path):
    from locus.db import (
        bootstrap_db,
        create_list,
        delete_source,
        get_source,
        write_source,
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path="/tmp/note.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    await delete_source("BV1abc")
    assert await get_source("BV1abc") is None


@pytest.mark.asyncio
async def test_get_sources_for_list(tmp_locus_home: Path):
    from locus.db import bootstrap_db, create_list, get_sources_for_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    for vid in ("BV1a", "BV1b"):
        await write_source(
            video_id=vid,
            platform="bilibili",
            list_id="list-1",
            title=vid,
            summary="",
            note_path=f"/tmp/{vid}.md",
            transcript_path=None,
            whisper_model="large-v3",
            ai_model="gpt-4o",
            vision_enabled=False,
            settings_snapshot={},
        )
    rows = await get_sources_for_list("list-1")
    assert len(rows) == 2


def test_locus_config_has_no_obsidian_field():
    from locus.config import LocusConfig

    cfg = LocusConfig()
    assert not hasattr(cfg, "obsidian")


def test_clear_embeddings_for_video_does_not_raise(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from locus.config import LocusConfig
    from locus.pipeline.embed import clear_embeddings_for_video

    mock_col = MagicMock()
    mock_col.delete.return_value = None
    with patch("locus.pipeline.embed._get_collection", return_value=mock_col):
        clear_embeddings_for_video("BV1abc", LocusConfig())
    mock_col.delete.assert_called_once_with(where={"note_id": "BV1abc"})


def test_write_video_note_to_locus_notes_dir(tmp_path: Path):
    from locus.adapters.base import VideoMeta
    from locus.pipeline.extract import ExtractionResult, KeyPoint
    from locus.pipeline.notes import write_video_note

    meta = VideoMeta(
        video_id="BV1test",
        title="Test Video",
        platform="bilibili",
        source_url="https://www.bilibili.com/video/BV1test",
        cover_url="",
        duration_seconds=120,
        uploader="User",
    )
    extraction = ExtractionResult(
        title="Test Video",
        summary="A great video.",
        key_points=[KeyPoint(timestamp="[00:00:01]", text="Key point")],
    )

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    with patch("locus.pipeline.notes.locus_home", return_value=tmp_path):
        note_path = write_video_note(meta, extraction, "list-1")

    assert note_path == notes_dir / "BV1test.md"
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "video_id: BV1test" in content
    assert "Test Video" in content
    assert "A great video." in content
