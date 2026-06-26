import json
from pathlib import Path

import pytest

from tests.factories import (
    ConversationFactory,
    MessageFactory,
    SectionedSourceFactory,
    SourceFactory,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_lists_table_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lists'")
        row = await cursor.fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_sources_table_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sources'")
        row = await cursor.fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_jobs_table_uses_meta_for_source_fields(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("PRAGMA table_info(jobs)")
        rows = await cursor.fetchall()
        columns = [row[1] for row in rows]

    assert "meta" in columns
    assert "source_url" not in columns
    assert "platform" not in columns


@pytest.mark.asyncio
async def test_sections_table_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sections'")
        row = await cursor.fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_sections_index_exists(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sections_source'")
        row = await cursor.fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_sections_table_columns(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("PRAGMA table_info(sections)")
        rows = await cursor.fetchall()
    columns = {row[1] for row in rows}
    assert columns == {
        "id",
        "source_id",
        "seq",
        "seg_start",
        "seg_end",
        "token_count",
        "timestamp_start",
        "timestamp_end",
        "summary",
        "keywords",
    }


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
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    source_id = "source-uuid-abc"
    await SourceFactory.build(
        "list-1",
        source_id=source_id,
        video_id="BV1abc",
        title="Intro to ML",
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
    )
    source = await get_source(source_id)
    assert source is not None
    assert source["title"] == "Intro to ML"


@pytest.mark.asyncio
async def test_delete_source(tmp_bibilab_home: Path):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        delete_source,
        get_source,
    )

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    source_id = "source-uuid-abc"
    await SourceFactory.build(
        "list-1",
        source_id=source_id,
        video_id="BV1abc",
        title="T",
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
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
    )

    await bootstrap_db()
    await create_list("list-1", "List 1", "2026-01-01T00:00:00")
    await create_list("list-2", "List 2", "2026-01-01T00:00:00")

    source_id_1 = "uuid-for-list-1"
    source_id_2 = "uuid-for-list-2"

    # Write same video to list-1
    await SourceFactory.build(
        "list-1",
        source_id=source_id_1,
        video_id="BV1abc",
        title="Video Title",
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
    )

    # Write same video to list-2 (should succeed, different source_id)
    await SourceFactory.build(
        "list-2",
        source_id=source_id_2,
        video_id="BV1abc",
        title="Video Title",
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
    )

    # Verify two rows exist (one per list)
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE video_id='BV1abc'")
        rows = await cursor.fetchall()
    assert len(rows) == 2
    source_ids = {row["id"] for row in rows}
    assert source_id_1 in source_ids
    assert source_id_2 in source_ids

    # Verify UNIQUE constraint via INSERT OR IGNORE - writing same video+list again
    # should not create a third row
    await SourceFactory.build(
        "list-1",
        source_id="uuid-should-be-ignored",
        video_id="BV1abc",
        title="Updated Title",
        source_url="https://bilibili.com/video/BV1abc",
        duration_seconds=600,
        uploader="Uploader",
        language="en",
        whisper_model="base",
    )

    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE video_id='BV1abc'")
        rows = await cursor.fetchall()
    assert len(rows) == 2  # Still only 2 rows
    # The first row (list-1) should be updated
    list1_row = next(row for row in rows if row["list_id"] == "list-1")
    assert list1_row["id"] == source_id_1  # source_id preserved
    assert list1_row["title"] == "Updated Title"  # but title updated


@pytest.mark.asyncio
async def test_get_sources_for_list(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_list, get_sources_for_list

    await bootstrap_db()
    await create_list("list-1", "ML Course", "2026-01-01T00:00:00")
    for i, vid in enumerate(("BV1a", "BV1b")):
        await SourceFactory.build(
            "list-1",
            source_id=f"source-uuid-{vid}",
            video_id=vid,
            title=vid,
            source_url=f"https://bilibili.com/video/{vid}",
            duration_seconds=600,
            uploader="Uploader",
            language="en",
        )
    rows = await get_sources_for_list("list-1")
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_get_video_statuses_empty(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    result = await get_video_statuses([], "list-1")
    assert result == {}


@pytest.mark.asyncio
async def test_get_video_statuses_all_new(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_list
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    result = await get_video_statuses(["BV1", "BV2", "BV3"], "list-1")
    assert result == {"BV1": "new", "BV2": "new", "BV3": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_all_processed(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_list
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    for i, vid in enumerate(("BV1", "BV2")):
        await SourceFactory.build(
            "list-1",
            source_id=f"src-{vid}",
            video_id=vid,
            title=f"Title {vid}",
            source_url=f"https://bilibili.com/video/{vid}",
            duration_seconds=600,
            uploader="Uploader",
            language="en",
            whisper_model="base",
        )
    result = await get_video_statuses(["BV1", "BV2"], "list-1")
    assert result == {"BV1": "processed", "BV2": "processed"}


@pytest.mark.asyncio
async def test_get_video_statuses_all_in_progress(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await create_job("ingest", {"video_id": "BV1", "list_id": "list-1"})
    await create_job("ingest", {"video_id": "BV2", "list_id": "list-1"})
    result = await get_video_statuses(["BV1", "BV2"], "list-1")
    assert result == {"BV1": "in_progress", "BV2": "in_progress"}


@pytest.mark.asyncio
async def test_get_video_statuses_all_needs_auth(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list, get_db
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await create_job("ingest", {"video_id": "BV1", "list_id": "list-1"})
    await create_job("ingest", {"video_id": "BV2", "list_id": "list-1"})
    async with get_db() as db:
        await db.execute("UPDATE jobs SET status='needs_auth' WHERE meta->>'$.video_id'='BV1'")
        await db.commit()
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "needs_auth"}


@pytest.mark.asyncio
async def test_get_video_statuses_mixed(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list, get_db
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await SourceFactory.build(
        "list-1",
        source_id="src-BV1",
        video_id="BV1",
        title="T1",
        source_url="url",
        duration_seconds=600,
        uploader="U",
        language="en",
        whisper_model="base",
    )
    await create_job("ingest", {"video_id": "BV2", "list_id": "list-1"})
    await create_job("ingest", {"video_id": "BV3", "list_id": "list-1"})
    async with get_db() as db:
        await db.execute("UPDATE jobs SET status='needs_auth' WHERE meta->>'$.video_id'='BV3'")
        await db.commit()
    result = await get_video_statuses(["BV1", "BV2", "BV3", "BV4"], "list-1")
    assert result == {"BV1": "processed", "BV2": "in_progress", "BV3": "needs_auth", "BV4": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_done_job_not_in_progress(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list, update_job_status
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    job_id = await create_job("ingest", {"video_id": "BV1", "list_id": "list-1"})
    await update_job_status(job_id, "done")
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_failed_is_new(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list, update_job_status
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    job_id = await create_job("ingest", {"video_id": "BV1", "list_id": "list-1"})
    await update_job_status(job_id, "failed")
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_job_list_id_isolation(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await create_list("list-2", "Test2", "2026-01-01T00:00:00")
    await create_job("ingest", {"video_id": "BV1", "list_id": "list-2"})
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_job_video_id_isolation(tmp_bibilab_home: Path) -> None:
    """A job for a different video_id in the same list must not affect the result."""
    from bibilab.db import bootstrap_db, create_job, create_list
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await create_job("ingest", {"video_id": "BV_other", "list_id": "list-1"})
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "new"}


@pytest.mark.asyncio
async def test_get_video_statuses_precedence_needs_auth_over_processed(tmp_bibilab_home: Path) -> None:
    from bibilab.db import bootstrap_db, create_job, create_list, get_db
    from bibilab.video_status import get_video_statuses

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await SourceFactory.build(
        "list-1",
        source_id="src-BV1",
        video_id="BV1",
        title="T1",
        source_url="url",
        duration_seconds=600,
        uploader="U",
        language="en",
        whisper_model="base",
    )
    await create_job("ingest", {"video_id": "BV1", "list_id": "list-1"})
    async with get_db() as db:
        await db.execute("UPDATE jobs SET status='needs_auth' WHERE meta->>'$.video_id'='BV1'")
        await db.commit()
    result = await get_video_statuses(["BV1"], "list-1")
    assert result == {"BV1": "needs_auth"}


def test_clear_embeddings_for_source_does_not_raise(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from bibilab.pipeline.embed import clear_embeddings_for_source

    mock_col = MagicMock()
    mock_col.delete.return_value = None
    with (
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_col),
        patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path),
    ):
        (tmp_path / "chroma").mkdir(parents=True, exist_ok=True)
        clear_embeddings_for_source("src-1")
    mock_col.delete.assert_called_once_with(where={"source_id": "src-1"})


def test_seg_range_exact_at_token_cap_boundary():
    """Seg-range exactly covers chunks at token-cap boundary (timestamp-trap).

    Adjacent chunks at a token-cap split are timestamp-contiguous — a
    time-range query would double-count the boundary segment. Seg-range
    (source-scoped seq) is exact: ranges tile [0, N-1] with no gap/overlap.
    Verified by driving chunk_segments directly — the stored range is tested
    in T4 manual smoke (fresh-DB re-ingest).
    """
    from bibilab.pipeline.chunk import chunk_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    segs = [WhisperSegment(start=float(i), end=float(i) + 1, text=f"句{i}。", speaker="S") for i in range(8)]
    chunks = chunk_segments(segs, target_tokens=2, chunk_max_tokens=3, language="zh")
    ranges = [(c.seg_start, c.seg_end) for c in chunks]
    assert ranges[0][0] == 0 and ranges[-1][1] == 7
    for (_, prev_end), (nxt_start, _) in zip(ranges, ranges[1:]):
        assert nxt_start == prev_end + 1  # contiguous, no gap/overlap


class TestDeriveVideoStatuses:
    def test_all_new_when_no_jobs_no_processed(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1", "v2"], [], set())
        assert result == {"v1": "new", "v2": "new"}

    def test_all_new_when_empty_video_ids(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses([], [], set())
        assert result == {}

    def test_processed_when_in_sources(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [], {"v1"})
        assert result == {"v1": "processed"}

    def test_processed_with_in_progress_job_shows_in_progress(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [{"video_id": "v1", "status": "queued"}]
        result = derive_video_statuses(["v1"], jobs, {"v1"})
        assert result == {"v1": "in_progress"}

    def test_in_progress_queued(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "queued"}], set())
        assert result == {"v1": "in_progress"}

    def test_in_progress_downloading(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "downloading"}], set())
        assert result == {"v1": "in_progress"}

    def test_in_progress_transcribing(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "transcribing"}], set())
        assert result == {"v1": "in_progress"}

    def test_in_progress_processing(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "processing"}], set())
        assert result == {"v1": "in_progress"}

    def test_in_progress_done_is_not_in_progress(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "done"}], set())
        assert result == {"v1": "new"}

    def test_in_progress_failed_is_not_in_progress(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "failed"}], set())
        assert result == {"v1": "new"}

    def test_needs_auth(self):
        from bibilab.video_status import derive_video_statuses

        result = derive_video_statuses(["v1"], [{"video_id": "v1", "status": "needs_auth"}], set())
        assert result == {"v1": "needs_auth"}

    def test_needs_auth_takes_precedence_over_in_progress(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [
            {"video_id": "v1", "status": "queued"},
            {"video_id": "v1", "status": "needs_auth"},
        ]
        result = derive_video_statuses(["v1"], jobs, set())
        assert result == {"v1": "needs_auth"}

    def test_in_progress_takes_precedence_over_processed_job(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [{"video_id": "v1", "status": "queued"}]
        result = derive_video_statuses(["v1"], jobs, {"v1"})
        assert result == {"v1": "in_progress"}

    def test_mixed_statuses(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [
            {"video_id": "v1", "status": "needs_auth"},
            {"video_id": "v2", "status": "downloading"},
            {"video_id": "v3", "status": "queued"},
        ]
        result = derive_video_statuses(["v1", "v2", "v3", "v4"], jobs, {"v3"})
        assert result == {
            "v1": "needs_auth",
            "v2": "in_progress",
            "v3": "in_progress",
            "v4": "new",
        }

    def test_job_for_different_video_not_considered(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [{"video_id": "other", "status": "queued"}]
        result = derive_video_statuses(["v1"], jobs, set())
        assert result == {"v1": "new"}

    def test_multiple_in_progress_jobs_same_video(self):
        from bibilab.video_status import derive_video_statuses

        jobs = [
            {"video_id": "v1", "status": "queued"},
            {"video_id": "v1", "status": "transcribing"},
        ]
        result = derive_video_statuses(["v1"], jobs, set())
        assert result == {"v1": "in_progress"}


@pytest.mark.asyncio
async def test_update_job_meta_merges_existing_keys(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, get_db, get_job, update_job_meta

    await bootstrap_db()
    await create_job("ingest", {"video_id": "BV1", "list_id": "list-1", "title": "Original"})
    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM jobs LIMIT 1")
        row = await cursor.fetchone()
        job_id = row["id"]

    await update_job_meta(job_id, {"source_id": "src-123"})

    job = await get_job(job_id)
    meta = json.loads(job["meta"])
    assert meta["video_id"] == "BV1"
    assert meta["title"] == "Original"
    assert meta["source_id"] == "src-123"


@pytest.mark.asyncio
async def test_update_job_meta_noops_on_missing_job(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, update_job_meta

    await bootstrap_db()
    await update_job_meta("nonexistent-id", {"source_id": "src-123"})


@pytest.mark.asyncio
async def test_message_tool_blocks_round_trip(tmp_bibilab_home: Path):
    """update_turn_terminal + get_recent_messages round-trip the tool_blocks JSON."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_recent_messages,
    )
    from bibilab.pipeline.chat_runs import update_turn_terminal

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    user_row = await MessageFactory.build(conv_id, role="user", status="pending")
    asst_row = await MessageFactory.build(conv_id, role="assistant", status="streaming")
    msg_id = asst_row["id"]

    blocks = [
        {
            "tool_use_id": "toolu_1",
            "name": "retrieve",
            "arguments": {"query": "q", "search_mode": "factual"},
            "result": {
                "chunks": [
                    {
                        "source_id": "s1",
                        "chunk_id": "v1_0_10",
                        "content": "x",
                        "video_title": "V1",
                        "timestamp_start": 0.0,
                        "timestamp_end": 10.0,
                        "citation_index": 1,
                    }
                ],
                "summary": {"sources_total": 1},
            },
        }
    ]

    await update_turn_terminal(
        conversation_id=conv_id,
        user_msg_id=user_row["id"],
        asst_msg_id=msg_id,
        asst_content="answer",
        asst_metadata=None,
        asst_tool_blocks=blocks,
        status="done",
        error=None,
    )

    rows = await get_recent_messages(conv_id, limit=10)
    asst = next(r for r in rows if r["id"] == msg_id)
    assert asst["tool_blocks"] is not None
    stored = json.loads(asst["tool_blocks"])
    assert stored == blocks


@pytest.mark.asyncio
async def test_update_turn_terminal_failed_leaves_user_error_null(tmp_bibilab_home: Path):
    """On a failed turn both rows flip to 'failed', but the error code lands only
    on the assistant row — the user message did not fail and must keep error=NULL."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_recent_messages,
    )
    from bibilab.pipeline.chat_runs import update_turn_terminal

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    user_row = await MessageFactory.build(conv_id, role="user", status="pending")
    asst_row = await MessageFactory.build(conv_id, role="assistant", status="streaming")

    await update_turn_terminal(
        conversation_id=conv_id,
        user_msg_id=user_row["id"],
        asst_msg_id=asst_row["id"],
        asst_content="",
        asst_metadata=None,
        asst_tool_blocks=None,
        status="failed",
        error="llm_rate_limit_error",
    )

    rows = {r["id"]: r for r in await get_recent_messages(conv_id, limit=10)}
    user_final, asst_final = rows[user_row["id"]], rows[asst_row["id"]]
    assert user_final["status"] == "failed"
    assert user_final["error"] is None
    assert asst_final["status"] == "failed"
    assert asst_final["error"] == "llm_rate_limit_error"


@pytest.mark.asyncio
async def test_get_source_facets(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_list, get_source_facets

    await bootstrap_db()
    await create_list("list-1", "Course", "2026-01-01T00:00:00")

    async def _write(sid: str, vid: str, seq, season):
        await SourceFactory.build(
            "list-1",
            source_id=sid,
            video_id=vid,
            title=vid,
            source_url="u",
            duration_seconds=1,
            uploader="u",
            language="en",
            whisper_model="w",
            ai_model="a",
            sequence_number=seq,
            season_number=season,
        )

    await _write("s1", "v1", 8, 2)
    await _write("s2", "v2", 9, None)
    await _write("s3", "v3", None, None)

    facets = await get_source_facets(["s1", "s2", "s3", "missing"])

    assert facets == {
        "s1": {"sequence_number": 8, "season_number": 2},
        "s2": {"sequence_number": 9, "season_number": None},
        "s3": {"sequence_number": None, "season_number": None},
    }
    assert await get_source_facets([]) == {}


# ── transcript_segments ──────────────────────────────────────────────


async def _insert_source(
    source_id: str = "src-1",
    video_id: str = "BV1",
    list_id: str = "list-1",
) -> str:
    from datetime import datetime, timezone

    from bibilab.db import create_list, get_db

    now = datetime.now(timezone.utc).isoformat()
    await create_list(list_id, "L", now)
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id) VALUES (?, ?, ?, ?)",
            (source_id, video_id, "bilibili", list_id),
        )
        await db.commit()
    return source_id


@pytest.mark.asyncio
async def test_transcript_segments_roundtrip(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_transcript_segments, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    source_id = await _insert_source()
    segs = [
        WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0"),
        WhisperSegment(start=2.0, end=4.0, text="再见。", speaker="SPK_1"),
    ]
    await write_transcript_segments(source_id, segs)
    rows = await get_transcript_segments(source_id)
    assert [r["seq"] for r in rows] == [0, 1]
    assert [r["text"] for r in rows] == ["你好。", "再见。"]
    assert [r["speaker"] for r in rows] == ["SPK_0", "SPK_1"]
    assert rows[0]["start_s"] == 0.0 and rows[1]["end_s"] == 4.0


@pytest.mark.asyncio
async def test_transcript_segments_cascade_on_source_delete(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, delete_source, get_transcript_segments, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    source_id = await _insert_source()
    await write_transcript_segments(source_id, [WhisperSegment(start=0.0, end=1.0, text="x。", speaker="SPK_0")])
    assert len(await get_transcript_segments(source_id)) == 1
    await delete_source(source_id)
    assert await get_transcript_segments(source_id) == []


@pytest.mark.asyncio
async def test_transcript_segments_rejects_orphan(tmp_bibilab_home: Path):
    import aiosqlite

    from bibilab.db import bootstrap_db, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    with pytest.raises(aiosqlite.IntegrityError):
        await write_transcript_segments(
            "no-such-source", [WhisperSegment(start=0.0, end=1.0, text="x。", speaker=None)]
        )


@pytest.mark.asyncio
async def test_get_segments_for_ranges_batches_multiple_sources(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_segments_for_ranges, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    s1 = await _insert_source("s1", "BV1", "list-1")
    s2 = await _insert_source("s2", "BV2", "list-2")
    await write_transcript_segments(
        s1,
        [
            WhisperSegment(start=0.0, end=1.0, text="a", speaker="SPK_0"),
            WhisperSegment(start=1.0, end=2.0, text="b", speaker="SPK_0"),
            WhisperSegment(start=2.0, end=3.0, text="c", speaker="SPK_1"),
        ],
    )
    await write_transcript_segments(
        s2,
        [
            WhisperSegment(start=0.0, end=1.0, text="d", speaker="SPK_0"),
        ],
    )

    rows = await get_segments_for_ranges([("s1", 1, 2), ("s2", 0, 0)])
    got = {(r["source_id"], r["seq"], r["text"]) for r in rows}
    assert got == {("s1", 1, "b"), ("s1", 2, "c"), ("s2", 0, "d")}


@pytest.mark.asyncio
async def test_get_segments_for_ranges_empty_returns_empty():
    from bibilab.db import get_segments_for_ranges

    assert await get_segments_for_ranges([]) == []


@pytest.mark.asyncio
async def test_sources_has_no_transcript_path_column(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("PRAGMA table_info(sources)")
        cols = [r["name"] for r in await cursor.fetchall()]
    assert "transcript_path" not in cols


_SOURCE_FIELDS = dict(
    source_id="src-1",
    video_id="BV1",
    platform="bilibili",
    list_id="list-1",
    title="T",
    cover_url=None,
    source_url="https://bilibili.com/video/BV1",
    duration_seconds=1,
    uploader="U",
    language="zh",
    whisper_model="m",
    ai_model="a",
    settings_snapshot={},
)


@pytest.mark.asyncio
async def test_write_source_with_segments_atomic_happy(tmp_bibilab_home: Path):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_source,
        get_transcript_segments,
        write_source_with_segments,
    )
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    await write_source_with_segments(
        segments=[WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0")],
        **_SOURCE_FIELDS,
    )
    assert await get_source("src-1") is not None
    assert len(await get_transcript_segments("src-1")) == 1


@pytest.mark.asyncio
async def test_write_source_with_segments_rolls_back_on_segment_failure(tmp_bibilab_home: Path, monkeypatch):
    """If the segment write fails, the source upsert rolls back with it — no
    orphaned source row (the bug the old compensating delete tried to patch)."""
    import bibilab.db as db
    from bibilab.db import bootstrap_db, create_list, get_source, write_source_with_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")

    async def _boom(*args, **kwargs):
        raise RuntimeError("segment write failed")

    monkeypatch.setattr(db, "_exec_write_transcript_segments", _boom)

    with pytest.raises(RuntimeError):
        await write_source_with_segments(
            segments=[WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0")],
            **_SOURCE_FIELDS,
        )
    assert await get_source("src-1") is None


# --- aborted turns invisible to LLM replay + compaction ---
# (get_message_count's done-only filter is covered by
# test_aborted_messages_do_not_trigger_compression in test_chat_summary.py)


@pytest.mark.asyncio
async def test_get_messages_beyond_window_excludes_non_done(tmp_bibilab_home: Path):
    """get_messages_beyond_window must not surface cancelled/failed rows to
    the summarizer — they would render as blank 'assistant:' lines."""
    from bibilab.db import bootstrap_db, create_list, get_messages_beyond_window

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    # Seed 12 rows with a cancelled pair in the middle (so it falls into the
    # old half beyond the 3-message window). Post-fix: both rows of the aborted
    # turn share the terminal status.
    await MessageFactory.build(conv_id, role="user", content="u0", status="done")
    await MessageFactory.build(conv_id, role="assistant", content="a0", status="done")
    await MessageFactory.build(conv_id, role="user", content="u1", status="done")
    await MessageFactory.build(conv_id, role="assistant", content="a1", status="done")
    await MessageFactory.build(conv_id, role="user", content="u-aborted", status="cancelled")
    await MessageFactory.build(conv_id, role="assistant", content="", status="cancelled")
    await MessageFactory.build(conv_id, role="user", content="u2", status="done")
    await MessageFactory.build(conv_id, role="assistant", content="a2", status="done")
    await MessageFactory.build(conv_id, role="user", content="u3", status="done")
    await MessageFactory.build(conv_id, role="assistant", content="a3", status="done")
    await MessageFactory.build(conv_id, role="user", content="u4", status="done")
    await MessageFactory.build(conv_id, role="assistant", content="a4", status="done")

    to_compress = await get_messages_beyond_window(conv_id, window_size=3)
    contents = [r["content"] for r in to_compress]
    statuses = [r["status"] for r in to_compress]
    # The cancelled assistant must not appear in the summary input.
    assert "" not in contents
    # The user row of the aborted turn must be excluded too (it has no
    # assistant partner after filtering, so it would orphan into the summary).
    assert "u-aborted" not in contents
    # All rows that DO appear must have status='done'.
    assert all(s == "done" for s in statuses)


@pytest.mark.asyncio
async def test_write_and_get_sections(tmp_bibilab_home: Path):
    from bibilab.db import (
        _exec_write_sections,
        bootstrap_db,
        create_list,
        get_db,
        get_sections,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1a")

    sections = [
        Section(seg_start=0, seg_end=10, token_count=5800, timestamp_start=0.0, timestamp_end=100.0),
        Section(seg_start=11, seg_end=22, token_count=6100, timestamp_start=101.0, timestamp_end=210.0),
    ]
    digests = [SectionDigest(summary=f"s{i}", keywords=[]) for i in range(len(sections))]
    async with get_db() as db:
        await _exec_write_sections(db, source_id, sections, digests)
        await db.commit()

    rows = await get_sections(source_id)
    assert len(rows) == 2
    assert rows[0]["seq"] == 0
    assert rows[0]["seg_start"] == 0
    assert rows[0]["seg_end"] == 10
    assert rows[0]["token_count"] == 5800
    assert rows[0]["timestamp_start"] == 0.0
    assert rows[0]["timestamp_end"] == 100.0
    assert rows[1]["seq"] == 1
    assert rows[1]["seg_start"] == 11


@pytest.mark.asyncio
async def test_write_sections_is_idempotent(tmp_bibilab_home: Path):
    from bibilab.db import (
        _exec_write_sections,
        bootstrap_db,
        create_list,
        get_db,
        get_sections,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1b")

    sections = [
        Section(seg_start=0, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
    ]
    digests = [SectionDigest(summary="s0", keywords=[])]
    async with get_db() as db:
        await _exec_write_sections(db, source_id, sections, digests)
        await db.commit()
    async with get_db() as db:
        await _exec_write_sections(db, source_id, sections, digests)  # re-ingest
        await db.commit()
    rows = await get_sections(source_id)
    assert len(rows) == 1  # DELETE+INSERT, not duplicate


@pytest.mark.asyncio
async def test_sections_cascade_on_source_delete(tmp_bibilab_home: Path):
    from bibilab.db import (
        _exec_write_sections,
        bootstrap_db,
        create_list,
        delete_source,
        get_db,
        get_sections,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1c")
    async with get_db() as db:
        await _exec_write_sections(
            db,
            source_id,
            [
                Section(seg_start=0, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
            ],
            [SectionDigest(summary="s0", keywords=[])],
        )
        await db.commit()
    await delete_source(source_id)
    assert await get_sections(source_id) == []


@pytest.mark.asyncio
async def test_write_source_with_segments_writes_sections_atomically(tmp_bibilab_home: Path):
    """write_source_with_segments must persist sections in the SAME
    transaction as the source row and segments — no orphan section rows on
    partial failure."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_sections,
        get_source,
        write_source_with_segments,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")

    sections = [
        Section(seg_start=0, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
    ]
    await write_source_with_segments(
        segments=[],
        sections=sections,
        section_digests=[SectionDigest(summary="s", keywords=[])],
        source_id="src-1",
        video_id="BV1x",
        platform="bilibili",
        list_id="list-1",
        title="T",
        cover_url=None,
        source_url="https://x",
        duration_seconds=0,
        uploader="u",
        language="en",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        settings_snapshot={},
    )

    assert await get_source("src-1") is not None
    rows = await get_sections("src-1")
    assert len(rows) == 1
    assert rows[0]["seg_start"] == 0


@pytest.mark.asyncio
async def test_write_source_with_segments_rollback_leaves_no_orphan_sections(
    tmp_bibilab_home: Path,
):
    """If _exec_write_sections raises, the parent source + segments must
    NOT commit. Verifies the FK cascade is not silently saving us — the
    transaction is rolled back as a unit."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_sections,
        get_source,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")

    sections = [
        Section(seg_start=0, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
    ]

    # Patch _exec_write_sections to raise mid-transaction; parent source
    # must NOT commit.
    from bibilab import db as db_mod

    original = db_mod._exec_write_sections

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated section-write failure")

    db_mod._exec_write_sections = boom
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            await db_mod.write_source_with_segments(
                segments=[],
                sections=sections,
                section_digests=[SectionDigest(summary="s", keywords=[])],
                source_id="src-fail",
                video_id="BV1fail",
                platform="bilibili",
                list_id="list-1",
                title="T",
                cover_url=None,
                source_url="https://x",
                duration_seconds=0,
                uploader="u",
                language="en",
                whisper_model="large-v3",
                ai_model="gpt-4o",
                settings_snapshot={},
            )
    finally:
        db_mod._exec_write_sections = original

    assert await get_source("src-fail") is None
    assert await get_sections("src-fail") == []


@pytest.mark.asyncio
async def test_write_source_with_segments_rollback_on_section_failure_also_rolls_back_segments(
    tmp_bibilab_home: Path,
):
    """Atomicity: a mid-transaction failure inside the sections writer
    must roll back the source row AND the transcript segments — proving the
    three-table write is one unit, not three independent inserts that happen
    to clean up after each other.

    Patches `_exec_write_sections` to raise after the segments are already
    staged. Post-conditions: no source row, no segments, no sections for
    the failing source_id.
    """
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_sections,
        get_source,
        get_transcript_segments,
        write_source_with_segments,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")

    segments = [WhisperSegment(start=float(i), end=float(i + 1), text=f"seg{i}.", speaker=None) for i in range(15)]
    sections = [Section(seg_start=0, seg_end=14, token_count=100, timestamp_start=0.0, timestamp_end=15.0)]
    section_digests = [SectionDigest(summary="S", keywords=["k"])]

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-write failure")

    import bibilab.db as db_mod

    original = db_mod._exec_write_sections
    db_mod._exec_write_sections = boom
    try:
        with pytest.raises(RuntimeError, match="simulated mid-write failure"):
            await write_source_with_segments(
                segments=segments,
                sections=sections,
                section_digests=section_digests,
                source_id="src-ac3",
                video_id="BV1ac3",
                platform="bilibili",
                list_id="list-1",
                title="T",
                cover_url=None,
                source_url="https://x",
                duration_seconds=15,
                uploader="u",
                language="en",
                whisper_model="x",
                ai_model="y",
                settings_snapshot={},
            )
    finally:
        db_mod._exec_write_sections = original

    assert await get_source("src-ac3") is None
    assert await get_sections("src-ac3") == []
    assert await get_transcript_segments("src-ac3") == []


@pytest.mark.asyncio
async def test_write_source_with_segments_sections_omitted_keeps_old_behavior(
    tmp_bibilab_home: Path,
):
    """Backwards compat: omitting `sections=` must still work (other call
    sites / tests don't pass it)."""
    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    await SourceFactory.build("list-1", source_id="src-legacy", video_id="BV1legacy")
    assert await get_source("src-legacy") is not None


# ── update_section_summaries ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_section_summaries_updates_existing_rows_by_seq(
    tmp_bibilab_home: Path,
):
    from bibilab.db import bootstrap_db, create_list, get_sections, update_section_summaries

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id, _segments, _sections, _digests = await SectionedSourceFactory.build("list-1", video_id="BV1sections")

    await update_section_summaries(
        source_id,
        [(0, "Summary 0", ["k0"]), (1, "Summary 1", ["k1"]), (2, "Summary 2", ["k2"])],
    )

    rows = await get_sections(source_id)
    assert len(rows) == 3
    assert rows[0]["summary"] == "Summary 0"
    assert json.loads(rows[0]["keywords"]) == ["k0"]
    assert rows[1]["summary"] == "Summary 1"
    assert rows[2]["summary"] == "Summary 2"


@pytest.mark.asyncio
async def test_update_section_summaries_missing_seq_is_silent_noop(
    tmp_bibilab_home: Path,
):
    """A seq not in the sections table is silently no-op'd by the UPDATE.

    The helper does not pre-validate: the caller's zip of
    `get_sections(source_id)` rows with the new digests guarantees unique
    existing seqs, so a missing seq here is a caller bug — but the
    database is the source of truth and the UPDATE simply doesn't match.
    """
    from bibilab.db import bootstrap_db, create_list, get_sections, update_section_summaries

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id, _segments, _sections, _digests = await SectionedSourceFactory.build("list-1", video_id="BV1sections2")

    # No raise; the missing seq 99 is silently no-op'd.
    await update_section_summaries(
        source_id,
        [
            (0, "S0", ["k"]),
            (1, "S1", ["k"]),
            (2, "S2", ["k"]),
            (99, "S99", ["k"]),  # seq 99 doesn't exist
        ],
    )
    rows = await get_sections(source_id)
    # seqs 0-2 are updated; seq 99 was a no-op (no row created).
    assert [r["summary"] for r in rows] == ["S0", "S1", "S2"]


@pytest.mark.asyncio
async def test_get_sections_returns_all_columns_ordered_by_seq(
    tmp_bibilab_home: Path,
):
    from bibilab.db import bootstrap_db, create_list, get_sections

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id, _segments, _sections, _digests = await SectionedSourceFactory.build("list-1", video_id="BV1sections")

    rows = await get_sections(source_id)
    assert len(rows) == 3
    # Ordered by seq.
    assert [r["seq"] for r in rows] == [0, 1, 2]
    # All 10 columns present.
    for col in (
        "id",
        "source_id",
        "seq",
        "seg_start",
        "seg_end",
        "token_count",
        "timestamp_start",
        "timestamp_end",
        "summary",
        "keywords",
    ):
        assert col in rows[0].keys(), f"missing column {col}"
    # Empty source returns empty list.
    assert await get_sections("nonexistent") == []


# ── write_source_with_segments: section_digests ingest path ──────────


@pytest.mark.asyncio
async def test_write_source_with_segments_section_digests_persists_summaries(
    tmp_bibilab_home: Path,
):
    """Atomic write with section_digests: section rows must have non-NULL
    summary/keywords, and the writes must be in the same transaction as
    source + segments (one call → all three land)."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_sections,
        get_source,
        write_source_with_segments,
    )
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # Factory path: build a source row to obtain a real source_id, then
    # read the full row back for the source fields the test needs.
    source_id = await SourceFactory.build("list-1", video_id="BV1digests")
    source = await get_source(source_id)
    assert source is not None

    segments = [WhisperSegment(start=float(i), end=float(i + 1), text=f"s{i}.", speaker=None) for i in range(30)]
    sections = [
        Section(seg_start=0, seg_end=9, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
        Section(seg_start=10, seg_end=19, token_count=100, timestamp_start=10.0, timestamp_end=20.0),
    ]
    section_digests = [
        SectionDigest(summary="Sum 0", keywords=["k0", "k1"]),
        SectionDigest(summary="Sum 1", keywords=["k2"]),
    ]
    await write_source_with_segments(
        segments=segments,
        sections=sections,
        section_digests=section_digests,
        source_id=source_id,
        video_id=source["video_id"],
        platform=source["platform"],
        list_id=source["list_id"],
        title=source["title"],
        cover_url=None,
        source_url=source["source_url"],
        duration_seconds=30,
        uploader=source["uploader"],
        language=source["language"],
        whisper_model="x",
        ai_model="y",
        settings_snapshot={},
    )

    rows = await get_sections(source_id)
    assert len(rows) == 2
    assert rows[0]["summary"] == "Sum 0"
    assert json.loads(rows[0]["keywords"]) == ["k0", "k1"]
    assert rows[1]["summary"] == "Sum 1"
    assert json.loads(rows[1]["keywords"]) == ["k2"]


@pytest.mark.asyncio
async def test_write_source_with_segments_sections_without_digests_raises(
    tmp_bibilab_home: Path,
):
    """Passing sections without section_digests is a misuse — every section
    row must carry a summary, so the write raises before any DB work."""
    from bibilab.db import bootstrap_db, create_list, write_source_with_segments
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")

    segments = [WhisperSegment(start=float(i), end=float(i + 1), text=f"s{i}.", speaker=None) for i in range(15)]
    sections = [Section(seg_start=0, seg_end=14, token_count=100, timestamp_start=0.0, timestamp_end=15.0)]
    with pytest.raises(ValueError, match="section_digests is required"):
        await write_source_with_segments(
            segments=segments,
            sections=sections,
            source_id="src-nodigests",
            video_id="BV1nodigests",
            platform="bilibili",
            list_id="list-1",
            title="T",
            cover_url=None,
            source_url="https://x",
            duration_seconds=15,
            uploader="u",
            language="en",
            whisper_model="x",
            ai_model="y",
            settings_snapshot={},
        )


@pytest.mark.asyncio
async def test_get_messages_beyond_window_orders_by_rowid_when_timestamps_collide(
    tmp_bibilab_home: Path,
):
    """8 messages inserted in the same transaction share `created_at`; the
    rowid tiebreaker must yield ascending insertion order across the window.
    """
    from bibilab.db import bootstrap_db, get_db, get_messages_beyond_window

    await bootstrap_db()

    shared_ts = "2026-06-25T12:00:00+00:00"
    conv_id = "test-conv-rowid"
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("test-list-rowid", "T", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO conversations (id, list_id, summary, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)",
            (conv_id, "test-list-rowid", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        for i in range(8):
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, NULL, ?)",
                (f"m-{i:02d}", conv_id, "user", f"M{i}", shared_ts),
            )
        await db.commit()

    beyond = await get_messages_beyond_window(conv_id, window_size=3)
    assert [r["id"] for r in beyond] == [f"m-{i:02d}" for i in range(5)]
