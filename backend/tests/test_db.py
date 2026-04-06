from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        yield tmp_path


@pytest.mark.asyncio
async def test_lists_table_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lists'")
        assert cur.fetchone() is not None


@pytest.mark.asyncio
async def test_sources_table_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sources'")
        assert cur.fetchone() is not None


@pytest.mark.asyncio
async def test_jobs_table_uses_meta_for_source_fields(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()]

    assert "meta" in columns
    assert "source_url" not in columns
    assert "platform" not in columns


@pytest.mark.asyncio
async def test_create_and_get_list(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_list, get_all_lists, get_list

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    row = await get_list("list-1")
    assert row is not None
    assert row["name"] == "ML Course"
    all_lists = await get_all_lists()
    assert len(all_lists) == 1


@pytest.mark.asyncio
async def test_write_and_get_source(tmp_bibilab_home: Path):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_source,
        source_exists,
        write_source,
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    transcript_file = tmp_bibilab_home / "transcripts" / "BV1abc.txt"
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("Intro to ML transcript.", encoding="utf-8")
    source_id = "source-uuid-abc"
    await write_source(
        source_id=source_id,
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="Intro to ML",
        summary="A great intro.",
        keywords=["ml", "intro"],
        cover_url=None,
        transcript_path=str(transcript_file),
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    source = await get_source(source_id)
    assert source is not None
    assert source["title"] == "Intro to ML"
    assert await source_exists("BV1abc", "list-1") is True
    assert await source_exists("BV1xyz", "list-1") is False


@pytest.mark.asyncio
async def test_delete_source(tmp_bibilab_home: Path):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        delete_source,
        get_source,
        write_source,
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    source_id = "source-uuid-abc"
    await write_source(
        source_id=source_id,
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    await delete_source(source_id)
    assert await get_source(source_id) is None


@pytest.mark.asyncio
async def test_sources_unique_constraint(tmp_bibilab_home: Path):
    """Test that (video_id, list_id) is unique - same video in two lists creates two rows."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_db,
        write_source,
    )

    await bootstrap_db()
    await create_list("list-1", "List 1", "2026-01-01T00:00:00")
    await create_list("list-2", "List 2", "2026-01-01T00:00:00")

    source_id_1 = "uuid-for-list-1"
    source_id_2 = "uuid-for-list-2"

    # Write same video to list-1
    await write_source(
        source_id=source_id_1,
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="Video Title",
        summary="Summary",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Write same video to list-2 (should succeed, different source_id)
    await write_source(
        source_id=source_id_2,
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-2",
        title="Video Title",
        summary="Summary",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Verify two rows exist (one per list)
    async with get_db() as db:
        rows = db.execute("SELECT * FROM sources WHERE video_id='BV1abc'").fetchall()
    assert len(rows) == 2
    source_ids = {row["id"] for row in rows}
    assert source_id_1 in source_ids
    assert source_id_2 in source_ids

    # Verify UNIQUE constraint via INSERT OR IGNORE - writing same video+list again
    # should not create a third row
    await write_source(
        source_id="uuid-should-be-ignored",
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="Updated Title",
        summary="Updated Summary",
        keywords=["new", "keywords"],
        cover_url=None,
        transcript_path=None,
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    async with get_db() as db:
        rows = db.execute("SELECT * FROM sources WHERE video_id='BV1abc'").fetchall()
    assert len(rows) == 2  # Still only 2 rows
    # The first row (list-1) should be updated
    list1_row = next(row for row in rows if row["list_id"] == "list-1")
    assert list1_row["id"] == source_id_1  # source_id preserved
    assert list1_row["title"] == "Updated Title"  # but title updated


@pytest.mark.asyncio
async def test_get_sources_for_list(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_list, get_sources_for_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    for i, vid in enumerate(("BV1a", "BV1b")):
        await write_source(
            source_id=f"source-uuid-{vid}",
            video_id=vid,
            platform="bilibili",
            list_id="list-1",
            title=vid,
            summary="",
            keywords=[],
            cover_url=None,
            transcript_path=None,
            source_url=f"https://bilibili.com/video/{vid}",
            duration_seconds=600,
            uploader="Uploader",
            language="en",
            whisper_model="large-v3",
            ai_model="gpt-4o",
            vision_enabled=False,
            settings_snapshot={},
        )
    rows = await get_sources_for_list("list-1")
    assert len(rows) == 2


def test_clear_embeddings_for_video_does_not_raise(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from bibilab.config import BibilabConfig
    from bibilab.pipeline.embed import clear_embeddings_for_video

    mock_col = MagicMock()
    mock_col.delete.return_value = None
    with patch("bibilab.pipeline.embed._get_collection", return_value=mock_col):
        clear_embeddings_for_video("BV1abc", BibilabConfig())
    mock_col.delete.assert_called_once_with(where={"video_id": "BV1abc"})
