import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.factories import ConversationFactory, MessageFactory, SourceFactory

pytestmark = pytest.mark.integration


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        yield tmp_path


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
        summary="A great intro.",
        keywords=["ml", "intro"],
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
        summary="S",
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
        summary="Summary",
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
        summary="Summary",
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
        summary="Updated Summary",
        keywords=["new", "keywords"],
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

    from bibilab.config import BibilabConfig
    from bibilab.pipeline.embed import clear_embeddings_for_source

    mock_col = MagicMock()
    mock_col.delete.return_value = None
    with (
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_col),
        patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path),
    ):
        (tmp_path / "chroma").mkdir(parents=True, exist_ok=True)
        clear_embeddings_for_source("src-1", BibilabConfig())
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
    """update_message_content + get_recent_messages round-trip the tool_blocks JSON."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_recent_messages,
        update_message_content,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    row = await MessageFactory.build(
        conv_id,
        role="assistant",
    )
    msg_id = row["id"]

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

    await update_message_content(
        msg_id,
        content="answer",
        metadata=None,
        status="done",
        error=None,
        tool_blocks=blocks,
    )

    rows = await get_recent_messages(conv_id, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_blocks"] is not None
    stored = json.loads(row["tool_blocks"])
    assert stored == blocks


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
            summary="s",
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
    summary="S",
    keywords=[],
    cover_url=None,
    source_url="https://bilibili.com/video/BV1",
    duration_seconds=1,
    uploader="U",
    language="zh",
    whisper_model="m",
    ai_model="a",
    vision_enabled=False,
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


# --- #403: aborted turns invisible to LLM replay + compaction ---


@pytest.mark.asyncio
async def test_get_message_count_counts_only_done(tmp_bibilab_home: Path):
    """get_message_count must count only done messages — aborted rows do not
    contribute to the >30 compression trigger."""
    from bibilab.db import bootstrap_db, create_list, get_message_count

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    for i in range(5):
        await MessageFactory.build(conv_id, role="user", content=f"u{i}", status="done")
        await MessageFactory.build(conv_id, role="assistant", content=f"a{i}", status="done")
    # Post-fix: aborted turn has matching terminal status on both rows.
    await MessageFactory.build(conv_id, role="user", content="u-aborted", status="cancelled")
    await MessageFactory.build(conv_id, role="assistant", content="", status="cancelled")

    assert await get_message_count(conv_id) == 10


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
