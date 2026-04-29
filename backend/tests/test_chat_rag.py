from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_query_chunks_empty_source_ids(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    result = await query_chunks("test query", [], cfg)

    assert result == []


@pytest.mark.asyncio
async def test_query_chunks_no_video_ids_found(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    with patch(
        "bibilab.pipeline.embed.get_video_ids_for_sources",
        new_callable=AsyncMock,
    ) as mock_map:
        mock_map.return_value = {}

        result = await query_chunks("test query", ["src-uuid-1", "src-uuid-2"], cfg)

        assert result == []
        mock_map.assert_called_once_with(["src-uuid-1", "src-uuid-2"])


@pytest.mark.asyncio
async def test_query_chunks_filters_by_video_id(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [
            [
                {
                    "video_id": "bvid123",
                    "video_title": "Test Video",
                    "timestamp_start": 0.0,
                    "timestamp_end": 10.0,
                }
            ]
        ],
        "distances": [[0.1]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch(
            "bibilab.pipeline.embed._get_collection",
            return_value=mock_collection,
        ) as mock_get_col,
    ):
        mock_map.return_value = {"src-uuid-1": "bvid123"}

        await query_chunks("test query", ["src-uuid-1"], cfg, top_k=5)

        mock_get_col.assert_called_once_with(cfg)
        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["query_texts"] == ["test query"]
        assert call_kwargs["n_results"] == 5
        assert call_kwargs["where"] == {"video_id": {"$in": ["bvid123"]}}


@pytest.mark.asyncio
async def test_query_chunks_applies_relevance_floor(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3", "c4"]],
        "metadatas": [
            [
                {"video_id": "bvid1", "video_title": "V1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "bvid2", "video_title": "V2", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "bvid3", "video_title": "V3", "timestamp_start": 10.0, "timestamp_end": 15.0},
                {"video_id": "bvid4", "video_title": "V4", "timestamp_start": 15.0, "timestamp_end": 20.0},
            ]
        ],
        "distances": [[0.15, 0.25, 0.28, 0.4]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "bvid1", "s2": "bvid2", "s3": "bvid3", "s4": "bvid4"}

        result = await query_chunks("test query", ["s1", "s2", "s3", "s4"], cfg)

        assert len(result) == 3
        distances = [c.distance for c in result]
        assert distances == [0.15, 0.25, 0.28]


@pytest.mark.asyncio
async def test_query_chunks_returns_chunk_metadata(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["some transcript content here"]],
        "metadatas": [
            [
                {
                    "video_id": "bvid999",
                    "video_title": "My Video Title",
                    "timestamp_start": 12.5,
                    "timestamp_end": 27.3,
                }
            ]
        ],
        "distances": [[0.15]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"src-abc": "bvid999"}

        result = await query_chunks("test query", ["src-abc"], cfg)

        assert len(result) == 1
        chunk = result[0]
        assert chunk.content == "some transcript content here"
        assert chunk.video_title == "My Video Title"
        assert chunk.timestamp_start == 12.5
        assert chunk.timestamp_end == 27.3
        assert chunk.video_id == "bvid999"
        assert chunk.distance == 0.15


@pytest.mark.asyncio
async def test_query_chunks_sorts_by_distance_ascending(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "V1", "timestamp_start": 0.0, "timestamp_end": 1.0},
                {"video_id": "v2", "video_title": "V2", "timestamp_start": 0.0, "timestamp_end": 1.0},
                {"video_id": "v3", "video_title": "V3", "timestamp_start": 0.0, "timestamp_end": 1.0},
            ]
        ],
        "distances": [[0.1, 0.2, 0.25]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "v1", "s2": "v2", "s3": "v3"}

        result = await query_chunks("test query", ["s1", "s2", "s3"], cfg)

        assert len(result) == 3
        assert result[0].distance == 0.1
        assert result[1].distance == 0.2
        assert result[2].distance == 0.25


@pytest.mark.asyncio
async def test_query_chunks_chroma_error_returns_empty(tmp_bibilab_home, caplog):
    import logging

    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.side_effect = Exception("ChromaDB connection error")

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        caplog.at_level(logging.WARNING),
    ):
        mock_map.return_value = {"src-1": "bvid1"}

        result = await query_chunks("test query", ["src-1"], cfg)

        assert result == []
        assert "ChromaDB query failed" in caplog.text


@pytest.mark.asyncio
async def test_get_video_ids_for_sources(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_db, get_video_ids_for_sources

    await bootstrap_db()

    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("list-1", "Test List", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-a", "bv123", "bilibili", "list-1", "Test", "https://example.com"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-b", "bv456", "bilibili", "list-1", "Test2", "https://example.com"),
        )
        await db.commit()

    result = await get_video_ids_for_sources(["src-a", "src-b", "src-c"])

    assert result == {"src-a": "bv123", "src-b": "bv456"}


@pytest.mark.asyncio
async def test_get_video_ids_for_sources_empty(tmp_bibilab_home):
    from bibilab.db import get_video_ids_for_sources

    result = await get_video_ids_for_sources([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_video_ids_for_sources_no_matches(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_video_ids_for_sources

    await bootstrap_db()

    result = await get_video_ids_for_sources(["nonexistent-src"])

    assert result == {}


# --- retrieve() wrapper tests ---


@pytest.mark.asyncio
async def test_retrieve_returns_result_with_metadata(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import RetrievalResult, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5, reranking_enabled=False))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk A", "chunk B", "chunk C"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v2", "video_title": "Video 2", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 10.0, "timestamp_end": 15.0},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "v1", "s2": "v2"}

        result = await retrieve("test query", ["s1", "s2"], cfg, top_k=5)

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 3
    assert result.mode == "focused"
    assert result.candidates_evaluated == 3
    assert result.sources_with_hits == 2
    assert result.sources_total == 2
    assert len(result.source_coverage) == 2
    assert result.source_coverage[0].video_id == "v1"
    # lower score = more relevant (stores -score after RRF/rerank)
    assert result.source_coverage[0].best_score == 0.1
    assert result.source_coverage[1].video_id == "v2"


@pytest.mark.asyncio
async def test_retrieve_empty_sources(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    result = await retrieve("test query", [], cfg)

    assert result.chunks == []
    assert result.sources_total == 0
    assert result.sources_with_hits == 0
    assert result.source_coverage == []


@pytest.mark.asyncio
async def test_retrieve_respects_mode_parameter(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    with patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map:
        mock_map.return_value = {}

        result = await retrieve("test query", ["s1"], cfg, mode="broad")

    assert result.mode == "broad"
    assert result.sources_total == 1


# --- Source-aware aggregation tests ---


@pytest.mark.asyncio
async def test_retrieve_broad_mode_keeps_best_per_source(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5, reranking_enabled=False))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3", "c4", "c5", "c6"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v2", "video_title": "Video 2", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v2", "video_title": "Video 2", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v3", "video_title": "Video 3", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v3", "video_title": "Video 3", "timestamp_start": 5.0, "timestamp_end": 10.0},
            ]
        ],
        "distances": [[0.05, 0.15, 0.10, 0.20, 0.12, 0.25]],
    }

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "v1", "s2": "v2", "s3": "v3"}

        result = await retrieve("test query", ["s1", "s2", "s3"], cfg, mode="broad")

    # Only best chunk per source
    assert len(result.chunks) == 3
    video_ids = [c.video_id for c in result.chunks]
    assert video_ids == ["v1", "v2", "v3"]
    # Best distances per source
    assert result.chunks[0].distance == 0.05
    assert result.chunks[1].distance == 0.10
    assert result.chunks[2].distance == 0.12
    # candidates_evaluated reflects all chunks before aggregation
    assert result.candidates_evaluated == 6


@pytest.mark.asyncio
async def test_retrieve_broad_mode_uses_candidate_pool(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import RETRIEVAL_CANDIDATE_POOL, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed.query_chunks", new_callable=AsyncMock) as mock_qc,
    ):
        mock_map.return_value = {"s1": "v1"}
        mock_qc.return_value = []

        await retrieve("test query", ["s1"], cfg, mode="broad", top_k=10)

        mock_qc.assert_called_once_with("test query", ["s1"], cfg, top_k=RETRIEVAL_CANDIDATE_POOL, video_ids=["v1"])


@pytest.mark.asyncio
async def test_retrieve_focused_mode_returns_all_chunks(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 10.0, "timestamp_end": 15.0},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "v1"}

        result = await retrieve("test query", ["s1"], cfg, mode="focused")

    # Focused mode keeps all chunks, not just best per source
    assert len(result.chunks) == 3
    assert all(c.video_id == "v1" for c in result.chunks)


@pytest.mark.asyncio
async def test_format_rag_context_broad_mode(tmp_bibilab_home):
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit
    from bibilab.routers.chat import _format_rag_context  # noqa: E402

    result = RetrievalResult(
        chunks=[
            RetrievedChunk(
                content="chunk about topic",
                video_title="Video A",
                timestamp_start=10.0,
                timestamp_end=20.0,
                video_id="v1",
                distance=0.1,
            ),
            RetrievedChunk(
                content="another chunk",
                video_title="Video B",
                timestamp_start=0.0,
                timestamp_end=5.0,
                video_id="v2",
                distance=0.2,
            ),
        ],
        mode="broad",
        candidates_evaluated=8,
        sources_with_hits=2,
        sources_total=5,
        source_coverage=[
            # lower score = more relevant (stores -score after RRF/rerank)
            SourceHit(video_id="v1", video_title="Video A", best_score=0.1),
            SourceHit(video_id="v2", video_title="Video B", best_score=0.2),
        ],
    )

    text = _format_rag_context(result, "my query")

    assert "Concept appears in 2 of 5 sources" in text
    assert "Best excerpt per source:" in text
    assert '[Video A @ 10s-20s]: "chunk about topic"' in text


# --- hybrid_search tests (issue #201) ---


def _make_chunk(
    content: str = "chunk",
    video_id: str = "v1",
    video_title: str = "Video 1",
    ts_start: float = 0.0,
    ts_end: float = 5.0,
    distance: float = 0.5,
):
    from bibilab.pipeline.embed import RetrievedChunk

    return RetrievedChunk(
        content=content,
        video_title=video_title,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        video_id=video_id,
        distance=distance,
    )


@pytest.mark.asyncio
async def test_hybrid_search_runs_fts_and_vector_in_parallel(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import hybrid_search

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0))

    vector_chunks = [_make_chunk(content="vec chunk", video_id="v1")]
    fts_chunks = [_make_chunk(content="fts chunk", video_id="v2")]

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
            return_value={"s1": "v1", "s2": "v2"},
        ),
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=vector_chunks,
        ) as mock_vector,
        patch(
            "bibilab.pipeline.embed.query_fts",
            new_callable=AsyncMock,
            return_value=fts_chunks,
        ) as mock_fts,
    ):
        result = await hybrid_search("test query", ["s1", "s2"], cfg, effective_top_k=30)

    mock_vector.assert_called_once_with("test query", ["s1", "s2"], cfg, top_k=30, video_ids=["v1", "v2"])
    mock_fts.assert_called_once_with("test query", ["s1", "s2"], cfg, top_k=30, video_ids=["v1", "v2"])
    assert len(result) == 2


@pytest.mark.asyncio
async def test_hybrid_search_fallback_when_fts_returns_empty(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import hybrid_search

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0))

    vector_chunks = [_make_chunk(content="vec chunk", video_id="v1")]

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
            return_value={"s1": "v1"},
        ),
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=vector_chunks,
        ) as _,
        patch(
            "bibilab.pipeline.embed.query_fts",
            new_callable=AsyncMock,
            return_value=[],
        ) as _,
    ):
        result = await hybrid_search("test query", ["s1"], cfg, effective_top_k=30)

    assert result == vector_chunks


@pytest.mark.asyncio
async def test_hybrid_search_fallback_when_fts_errors(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import hybrid_search

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0))

    vector_chunks = [_make_chunk(content="vec chunk", video_id="v1")]

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
            return_value={"s1": "v1"},
        ),
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=vector_chunks,
        ) as _,
        patch(
            "bibilab.pipeline.embed.query_fts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("FTS error"),
        ) as _,
    ):
        result = await hybrid_search("test query", ["s1"], cfg, effective_top_k=30)

    assert result == vector_chunks


@pytest.mark.asyncio
async def test_hybrid_search_deduplicates_same_chunk(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import hybrid_search

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0))

    same_chunk = _make_chunk(
        content="same chunk",
        video_id="v1",
        video_title="Video 1",
        ts_start=10.0,
        ts_end=20.0,
        distance=0.3,
    )
    vector_chunks = [same_chunk]
    fts_chunks = [same_chunk]

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
            return_value={"s1": "v1"},
        ),
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=vector_chunks,
        ),
        patch(
            "bibilab.pipeline.embed.query_fts",
            new_callable=AsyncMock,
            return_value=fts_chunks,
        ),
    ):
        result = await hybrid_search("test query", ["s1"], cfg, effective_top_k=30)

    assert len(result) == 1
    assert result[0].content == "same chunk"


@pytest.mark.asyncio
async def test_retrieve_uses_hybrid_search(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import RETRIEVAL_CANDIDATE_POOL, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False))

    chunks = [_make_chunk(content="a", video_id="v1")]

    with patch(
        "bibilab.pipeline.embed.hybrid_search",
        new_callable=AsyncMock,
        return_value=chunks,
    ) as mock_hybrid:
        result = await retrieve("query", ["src1"], cfg, top_k=5)

    mock_hybrid.assert_called_once_with("query", ["src1"], cfg, effective_top_k=RETRIEVAL_CANDIDATE_POOL)
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_retrieve_hybrid_disabled_skips_fts(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import RETRIEVAL_CANDIDATE_POOL, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, hybrid_enabled=False, reranking_enabled=False))

    chunks = [_make_chunk(content="a", video_id="v1")]

    with (
        patch(
            "bibilab.pipeline.embed.hybrid_search",
            new_callable=AsyncMock,
        ) as mock_hybrid,
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ) as mock_vector,
    ):
        result = await retrieve("query", ["src1"], cfg, top_k=5)

    mock_hybrid.assert_not_called()
    mock_vector.assert_called_once_with("query", ["src1"], cfg, top_k=RETRIEVAL_CANDIDATE_POOL)
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_retrieve_focused_mode_uses_candidate_pool_before_rerank(tmp_bibilab_home):
    """Focused mode must pull RETRIEVAL_CANDIDATE_POOL candidates, not top_k, so that
    the cross-encoder reranker has a meaningful pool to trim from."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import RETRIEVAL_CANDIDATE_POOL, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True))

    candidate_chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}") for i in range(20)]
    reranked = candidate_chunks[:5]

    with (
        patch(
            "bibilab.pipeline.embed.hybrid_search",
            new_callable=AsyncMock,
            return_value=candidate_chunks,
        ) as mock_hybrid,
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=reranked,
        ) as mock_rerank,
    ):
        result = await retrieve("query", ["src1"], cfg, mode="focused", top_k=5)

    mock_hybrid.assert_called_once_with("query", ["src1"], cfg, effective_top_k=RETRIEVAL_CANDIDATE_POOL)
    mock_rerank.assert_called_once_with("query", candidate_chunks, top_k=5)
    assert result.chunks == reranked
    assert result.candidates_evaluated == len(candidate_chunks)


def test_rrf_fuse_ranks_doc_in_both_lists_above_doc_in_one():
    from bibilab.pipeline.embed import _rrf_fuse

    doc_a = _make_chunk(content="a", video_id="v1", distance=0.1)
    doc_b = _make_chunk(content="b", video_id="v2", distance=0.2)
    doc_c = _make_chunk(content="c", video_id="v3", distance=0.3)

    vec_list = [doc_a, doc_b, doc_c]
    fts_list = [doc_a, doc_c]

    result = _rrf_fuse(vec_list, fts_list, k=60)

    assert result[0].content == "a"
    assert result.index(doc_c) < result.index(doc_b)


@pytest.mark.asyncio
async def test_retrieve_broad_mode_keeps_highest_scoring_chunk_per_source(tmp_bibilab_home):
    """Regression: when chunks have .score set, broad mode must keep the highest-scoring
    chunk per source, not the lowest. _score() must negate score so that ascending sort
    correctly puts most-relevant first."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False))

    v1_low = _make_chunk(
        content="v1 low",
        video_id="v1",
        video_title="Video 1",
        ts_start=0.0,
        ts_end=5.0,
        distance=0.1,
    )
    v1_high = _make_chunk(
        content="v1 high",
        video_id="v1",
        video_title="Video 1",
        ts_start=10.0,
        ts_end=15.0,
        distance=0.2,
    )
    v2_low = _make_chunk(
        content="v2 low",
        video_id="v2",
        video_title="Video 2",
        ts_start=0.0,
        ts_end=5.0,
        distance=0.1,
    )
    v2_high = _make_chunk(
        content="v2 high",
        video_id="v2",
        video_title="Video 2",
        ts_start=10.0,
        ts_end=15.0,
        distance=0.3,
    )

    v1_low.score = 0.2
    v1_high.score = 0.9
    v2_low.score = 0.3
    v2_high.score = 0.7

    chunks = [v1_low, v1_high, v2_low, v2_high]

    with patch(
        "bibilab.pipeline.embed.hybrid_search",
        new_callable=AsyncMock,
        return_value=chunks,
    ):
        result = await retrieve("query", ["s1", "s2"], cfg, mode="broad", top_k=5)

    assert len(result.chunks) == 2
    chunk_by_video = {c.video_id: c for c in result.chunks}
    assert chunk_by_video["v1"].content == "v1 high"
    assert chunk_by_video["v2"].content == "v2 high"

    assert result.source_coverage[0].video_id == "v1"
    assert result.source_coverage[1].video_id == "v2"


@pytest.mark.asyncio
async def test_candidates_evaluated_reflects_pre_rerank_count(tmp_bibilab_home):
    """candidates_evaluated equals chunk count BEFORE rerank trim, not after."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3", "c4", "c5"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v1", "video_title": "Video 1", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v2", "video_title": "Video 2", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "v2", "video_title": "Video 2", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "v3", "video_title": "Video 3", "timestamp_start": 0.0, "timestamp_end": 5.0},
            ]
        ],
        "distances": [[0.05, 0.15, 0.10, 0.20, 0.12]],
    }

    reranked_chunks = [
        _make_chunk(content="c1", video_id="v1", ts_start=0.0, ts_end=5.0),
        _make_chunk(content="c2", video_id="v1", ts_start=5.0, ts_end=10.0),
    ]

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        patch("bibilab.pipeline.rerank.rerank", new_callable=AsyncMock, return_value=reranked_chunks),
    ):
        mock_map.return_value = {"s1": "v1", "s2": "v2", "s3": "v3"}

        result = await retrieve("test query", ["s1", "s2", "s3"], cfg, mode="focused", top_k=2)

    # candidates_evaluated should reflect the 5 chunks retrieved BEFORE reranking trimmed to top_k=2
    assert result.candidates_evaluated == 5
    # But chunks returned should be the top_k after reranking
    assert len(result.chunks) == 2
