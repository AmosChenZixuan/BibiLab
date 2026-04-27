"""Tests for the SQLite FTS5 full-text search index."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, _reset_cache
from bibilab.db import (
    bootstrap_db,
    clear_fts_for_video,
    get_db,
    query_fts_rows,
)
from bibilab.pipeline.chunk import RagChunk
from bibilab.pipeline.embed import populate_fts, query_fts


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    _reset_cache()
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.db.get_db_path", return_value=tmp_path / "bibilab.db"):
            with patch("bibilab.pipeline.embed.get_db_path", return_value=tmp_path / "bibilab.db"):
                yield tmp_path


def _make_meta(video_id: str = "VID1", title: str = "Test Video") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        platform="test",
        source_url="http://example.com",
        cover_url="",
        duration_seconds=60,
        uploader="tester",
    )


def _make_chunks(texts: list[str]) -> list[RagChunk]:
    return [
        RagChunk(text=t, timestamp_start=float(i * 10), timestamp_end=float(i * 10 + 9), sequence_index=i)
        for i, t in enumerate(texts)
    ]


@pytest.mark.asyncio
async def test_fts_table_created_on_bootstrap(tmp_bibilab_home: Path):
    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_populate_fts_inserts_rows(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta = _make_meta()
    chunks = _make_chunks(["hello world", "foo bar baz"])
    populate_fts(chunks, meta)

    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts")
        row = await cursor.fetchone()
    assert row[0] == 2


@pytest.mark.asyncio
async def test_query_fts_returns_matching_chunks(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta = _make_meta()
    chunks = _make_chunks(["the quick brown fox", "lazy dog sleeping", "fox jumps over fence"])
    populate_fts(chunks, meta)

    rows = await query_fts_rows("fox", ["VID1"], top_k=10)
    assert len(rows) >= 1
    contents = [r["content"] for r in rows]
    assert any("fox" in c for c in contents)


@pytest.mark.asyncio
async def test_query_fts_filters_by_video_id(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta1 = _make_meta("VID1", "Video One")
    meta2 = _make_meta("VID2", "Video Two")
    populate_fts(_make_chunks(["alpha beta gamma"]), meta1)
    populate_fts(_make_chunks(["alpha delta epsilon"]), meta2)

    rows = await query_fts_rows("alpha", ["VID1"], top_k=10)
    assert len(rows) == 1
    assert rows[0]["video_id"] == "VID1"

    rows_both = await query_fts_rows("alpha", ["VID1", "VID2"], top_k=10)
    assert len(rows_both) == 2


@pytest.mark.asyncio
async def test_query_fts_empty_sources_returns_empty(tmp_bibilab_home: Path):
    await bootstrap_db()
    rows = await query_fts_rows("anything", [], top_k=10)
    assert rows == []


@pytest.mark.asyncio
async def test_clear_fts_for_video(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta = _make_meta()
    populate_fts(_make_chunks(["some text here"]), meta)

    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts")
        assert (await cursor.fetchone())[0] == 1

    await clear_fts_for_video("VID1")

    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts")
        assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_query_fts_bm25_ranking(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta = _make_meta()
    chunks = _make_chunks(
        [
            "cat dog bird",
            "cat cat cat cat cat",  # more occurrences of "cat"
            "fish whale shark",
        ]
    )
    populate_fts(chunks, meta)

    rows = await query_fts_rows("cat", ["VID1"], top_k=10)
    assert len(rows) == 2
    # FTS5 rank: more negative = better match. First result should be the better match.
    assert rows[0]["rank"] <= rows[1]["rank"]


@pytest.mark.asyncio
async def test_query_fts_integration(tmp_bibilab_home: Path):
    """Test query_fts (the high-level async function) with mocked source lookup."""
    await bootstrap_db()
    meta = _make_meta()
    populate_fts(_make_chunks(["machine learning is great", "deep learning rocks"]), meta)

    with patch("bibilab.pipeline.embed.get_video_ids_for_sources", return_value={"src1": "VID1"}):
        results = await query_fts("learning", ["src1"], BibilabConfig())

    assert len(results) == 2
    assert all(r.distance > 0 for r in results)  # negated rank should be positive
    assert all("learning" in r.content for r in results)
