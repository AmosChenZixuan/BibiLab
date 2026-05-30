"""Tests for the SQLite FTS5 full-text search index."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.config import BibilabConfig, _reset_cache
from bibilab.db import (
    _cjk_bigram_tokens,
    _cjk_query_tokens,
    _cjk_runs,
    _escape_fts_query,
    _tokenize_cjk,
    bootstrap_db,
    get_db,
    query_fts_rows,
)
from bibilab.pipeline.chunk import RagChunk
from bibilab.pipeline.embed import clear_fts_for_video_sync, populate_fts, query_fts


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
        RagChunk(
            text=t,
            timestamp_start=float(i * 10),
            timestamp_end=float(i * 10 + 9),
            sequence_index=i,
            seg_start=i,
            seg_end=i,
        )
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

    await asyncio.to_thread(clear_fts_for_video_sync, "VID1")

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
    assert all(r.score is not None and r.score > 0 for r in results)  # negated BM25 rank; higher = more relevant
    assert all("learning" in r.content for r in results)


def test_escape_fts_query_empty_string():
    assert _escape_fts_query("") == ""


def test_escape_fts_query_only_whitespace():
    assert _escape_fts_query("   \t  ") == ""


def test_escape_fts_query_fts_operators():
    escaped = _escape_fts_query("AND OR NOT * ^")
    assert escaped == '"AND" "OR" "NOT" "*" "^"'


def test_escape_fts_query_preserves_normal_tokens():
    escaped = _escape_fts_query("machine learning")
    assert escaped == '"machine" "learning"'


def test_escape_fts_query_embedded_quotes():
    escaped = _escape_fts_query('say "hello world"')
    assert escaped == '"say" """hello" "world"""'


# --- _cjk_runs ---


def test_cjk_runs_all_cjk():
    runs = list(_cjk_runs("面食做法"))
    assert runs == [(True, "面食做法")]


def test_cjk_runs_all_non_cjk():
    runs = list(_cjk_runs("hello world"))
    assert runs == [(False, "hello world")]


def test_cjk_runs_mixed():
    runs = list(_cjk_runs("103岁生日这天"))
    assert runs == [(False, "103"), (True, "岁生日这天")]


def test_cjk_runs_multiple_cjk_segments():
    runs = list(_cjk_runs("第5集 女巫"))
    assert runs == [(True, "第"), (False, "5"), (True, "集"), (False, " "), (True, "女巫")]


# --- _cjk_bigram_tokens ---


def test_cjk_bigram_tokens_cjk_only():
    assert _cjk_bigram_tokens("面食做法") == ["面", "食", "做", "法", "面食", "食做", "做法"]


def test_cjk_bigram_tokens_two_chars():
    assert _cjk_bigram_tokens("面食") == ["面", "食", "面食"]


def test_cjk_bigram_tokens_single_cjk_char():
    assert _cjk_bigram_tokens("面") == ["面"]


def test_cjk_bigram_tokens_mixed():
    assert _cjk_bigram_tokens("103岁生日这天") == ["103", "岁", "生", "日", "这", "天", "岁生", "生日", "日这", "这天"]


def test_cjk_bigram_tokens_non_cjk():
    assert _cjk_bigram_tokens("hello world") == ["hello", "world"]


# --- _cjk_query_tokens (bigrams-only for multi-char, unigrams for single-char) ---


def test_cjk_query_tokens_multi_char_cjk():
    assert _cjk_query_tokens("面食做法") == ["面食", "食做", "做法"]


def test_cjk_query_tokens_two_chars():
    assert _cjk_query_tokens("面食") == ["面食"]


def test_cjk_query_tokens_single_cjk_char():
    assert _cjk_query_tokens("面") == ["面"]


def test_cjk_query_tokens_mixed():
    assert _cjk_query_tokens("103岁生日这天") == ["103", "岁生", "生日", "日这", "这天"]


def test_cjk_query_tokens_non_cjk():
    assert _cjk_query_tokens("hello world") == ["hello", "world"]


# --- _tokenize_cjk (unigram + bigram) ---


def test_tokenize_cjk_unigram_bigram_cjk_phrase():
    assert _tokenize_cjk("面食做法") == "面 食 做 法 面食 食做 做法"


def test_tokenize_cjk_unigram_bigram_two_chars():
    assert _tokenize_cjk("面食") == "面 食 面食"


def test_tokenize_cjk_preserves_non_cjk():
    assert _tokenize_cjk("ABC 123 hello") == "ABC 123 hello"


def test_tokenize_cjk_mixed_text():
    assert _tokenize_cjk("103岁生日这天") == "103 岁 生 日 这 天 岁生 生日 日这 这天"


def test_tokenize_cjk_normalizes_segment_boundary_spaces():
    # chunk.py:51 joins segments with " " → CJK-adjacent spaces must be
    # collapsed so bigrams can span the boundary
    assert _tokenize_cjk("第五 集") == "第 五 集 第五 五集"


def test_tokenize_cjk_empty_string():
    assert _tokenize_cjk("") == ""


# --- _escape_fts_query with CJK unigram + bigram ---


def test_escape_fts_query_chinese_phrase():
    # Multi-char CJK → bigrams only (no unigrams) to avoid AND-term explosion
    escaped = _escape_fts_query("面食做法")
    assert escaped == '"面食" "食做" "做法"'


def test_escape_fts_query_chinese_two_words():
    # Per-word bigram: "女巫" + "生日" → no cross-boundary "巫生"
    escaped = _escape_fts_query("女巫 生日")
    assert escaped == '"女巫" "生日"'


def test_escape_fts_query_chinese_no_whitespace_single_word():
    # No space → one word → bigrams across the whole string
    escaped = _escape_fts_query("女巫生日")
    assert escaped == '"女巫" "巫生" "生日"'


def test_escape_fts_query_single_cjk_word_three_chars():
    escaped = _escape_fts_query("多少岁")
    assert escaped == '"多少" "少岁"'


def test_escape_fts_query_single_cjk_char_preserved():
    # "茶" is a single-char word → unigram preserved
    escaped = _escape_fts_query("女巫 茶")
    assert escaped == '"女巫" "茶"'


def test_escape_fts_query_single_cjk_char_alone():
    escaped = _escape_fts_query("茶")
    assert escaped == '"茶"'


def test_escape_fts_query_high_idf_single_char():
    # 死 (death) — single-char word with critical semantic weight
    escaped = _escape_fts_query("死")
    assert escaped == '"死"'
