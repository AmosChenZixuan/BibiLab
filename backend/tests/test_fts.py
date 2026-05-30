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
    clear_fts_for_list,
    get_db,
    query_fts_rows,
)
from bibilab.pipeline.chunk import RagChunk
from bibilab.pipeline.embed import clear_fts_for_source_sync, populate_fts, query_fts


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
    populate_fts(chunks, "SRC1", meta)

    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts")
        row = await cursor.fetchone()
    assert row[0] == 2


@pytest.mark.asyncio
async def test_query_fts_returns_matching_chunks(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta = _make_meta()
    chunks = _make_chunks(["the quick brown fox", "lazy dog sleeping", "fox jumps over fence"])
    populate_fts(chunks, "SRC1", meta)

    rows = await query_fts_rows("fox", ["SRC1"], top_k=10)
    assert len(rows) >= 1
    contents = [r["content"] for r in rows]
    assert any("fox" in c for c in contents)


@pytest.mark.asyncio
async def test_query_fts_filters_by_source_id(tmp_bibilab_home: Path):
    await bootstrap_db()
    meta1 = _make_meta("VID1", "Video One")
    meta2 = _make_meta("VID2", "Video Two")
    populate_fts(_make_chunks(["alpha beta gamma"]), "S1", meta1)
    populate_fts(_make_chunks(["alpha delta epsilon"]), "S2", meta2)

    rows = await query_fts_rows("alpha", ["S1"], top_k=10)
    assert len(rows) == 1
    assert rows[0]["source_id"] == "S1"

    rows_both = await query_fts_rows("alpha", ["S1", "S2"], top_k=10)
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
    populate_fts(_make_chunks(["some text here"]), "SRC1", meta)

    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM chunks_fts")
        assert (await cursor.fetchone())[0] == 1

    await asyncio.to_thread(clear_fts_for_source_sync, "SRC1")

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
    populate_fts(chunks, "SRC1", meta)

    rows = await query_fts_rows("cat", ["SRC1"], top_k=10)
    assert len(rows) == 2
    # FTS5 rank: more negative = better match. First result should be the better match.
    assert rows[0]["rank"] <= rows[1]["rank"]


@pytest.mark.asyncio
async def test_query_fts_integration(tmp_bibilab_home: Path):
    """Test query_fts (the high-level async function) with mocked source lookup."""
    await bootstrap_db()
    meta = _make_meta()
    populate_fts(_make_chunks(["machine learning is great", "deep learning rocks"]), "SRC1", meta)

    results = await query_fts("learning", ["SRC1"], BibilabConfig())

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


# --- P3: source_id keying behavioral tests ---


@pytest.mark.asyncio
async def test_two_sources_same_video_id_independent_scoping(tmp_bibilab_home: Path):
    """#379: Two sources sharing one video_id scope independently.

    Under the old video_id-based keying, re-embedding source B could overwrite
    source A's Chroma data, and retrieval for one list could return chunks from
    another.  With source_id keying, each source is isolated even when both
    reference the same platform video_id.
    """
    await bootstrap_db()

    # Same video_id for both sources — the key regression scenario
    shared_video_id = "BV123"
    meta_a = _make_meta(shared_video_id, "Video Shared")
    meta_b = _make_meta(shared_video_id, "Video Shared")

    populate_fts(_make_chunks(["alpha only in list one"]), "src-a", meta_a)
    populate_fts(_make_chunks(["beta only in list two"]), "src-b", meta_b)

    # Query src-a — must return only src-a's chunk
    rows_a = await query_fts_rows("alpha", ["src-a"], top_k=10)
    assert len(rows_a) == 1
    assert rows_a[0]["source_id"] == "src-a"
    assert "alpha only in list one" in rows_a[0]["content"]

    # Query src-b — must return only src-b's chunk
    rows_b = await query_fts_rows("beta", ["src-b"], top_k=10)
    assert len(rows_b) == 1
    assert rows_b[0]["source_id"] == "src-b"
    assert "beta only in list two" in rows_b[0]["content"]

    # Cross-query: src-a should not see src-b's content
    rows_cross = await query_fts_rows("beta", ["src-a"], top_k=10)
    assert len(rows_cross) == 0


@pytest.mark.asyncio
async def test_clear_fts_for_list_independent_per_list(tmp_bibilab_home: Path):
    """#379: Deleting one list only clears FTS for its own sources.

    Under the old video_id-keyed query (WHERE video_id IN (SELECT video_id FROM
    sources WHERE list_id = ?)), deleting list 1 would also purge FTS rows for
    list 2's sources when both lists contained the same video_id.  The source_id
    subquery (WHERE source_id IN (SELECT id FROM sources WHERE list_id = ?))
    fixes this by keying on the source UUID directly.
    """
    await bootstrap_db()

    shared_video_id = "BV_SHARED"
    meta = _make_meta(shared_video_id, "Shared Video")

    # Create two lists, each with one source pointing to the same video_id
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("list-1", "List One", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("list-2", "List Two", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-list1", shared_video_id, "bilibili", "list-1", "T1", "http://x.com"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-list2", shared_video_id, "bilibili", "list-2", "T2", "http://x.com"),
        )
        await db.commit()

    populate_fts(_make_chunks(["list one chunk"]), "src-list1", meta)
    populate_fts(_make_chunks(["list two chunk"]), "src-list2", meta)

    # Verify both sets exist
    rows_all = await query_fts_rows("chunk", ["src-list1", "src-list2"], top_k=10)
    assert len(rows_all) == 2

    # Delete list 1 — should only clear src-list1's FTS rows
    await clear_fts_for_list("list-1")

    # List 1's data gone
    rows_list1 = await query_fts_rows("chunk", ["src-list1"], top_k=10)
    assert len(rows_list1) == 0

    # List 2's data still intact
    rows_list2 = await query_fts_rows("chunk", ["src-list2"], top_k=10)
    assert len(rows_list2) == 1
    assert rows_list2[0]["source_id"] == "src-list2"
