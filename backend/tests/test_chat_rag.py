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
    from bibilab.models._enums import RetrievalParams
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

        result = await retrieve("test query", ["s1", "s2"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 3
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
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    result = await retrieve("test query", [], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert result.chunks == []
    assert result.sources_total == 0
    assert result.sources_with_hits == 0
    assert result.source_coverage == []


@pytest.mark.asyncio
async def test_retrieve_single_source_returns_top_k_chunks(tmp_bibilab_home):
    """When LLM scopes to one source (#287), retrieve must return up to top_k
    chunks from that source, not be capped at spec_depth=2."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5, reranking_enabled=False, hybrid_enabled=False))

    # Seed 6 chunks from a single video via mock collection
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [[f"chunk{i} content" for i in range(6)]],
        "metadatas": [
            [
                {
                    "video_id": "v1",
                    "video_title": "Ramen Video",
                    "timestamp_start": float(i) * 10,
                    "timestamp_end": float(i) * 10 + 9.9,
                }
                for i in range(6)
            ]
        ],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.5]],
    }

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"source-1": "v1"}

        result = await retrieve(
            query_text="ramen",
            source_ids=["source-1"],
            cfg=cfg,
            params=RetrievalParams(depth_per_source=2, top_k=8),
            scoped_source_ids=None,
        )

    # Pre-fix: returns 2. Post-fix: returns up to 6 (all available) because
    # _adaptive_depth(2, 8, 1) = 8 → single source gets depth=8.
    assert len(result.chunks) == 6, f"expected 6 chunks, got {len(result.chunks)}"


# --- Source-aware aggregation tests ---


@pytest.mark.asyncio
async def test_diverse_top_k_depth_one_keeps_best_per_source(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
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

        result = await retrieve(
            "test query", ["s1", "s2", "s3"], cfg, params=RetrievalParams(depth_per_source=1, top_k=3)
        )

    # Only best chunk per source (depth=1, ranked by distance ascending)
    assert len(result.chunks) == 3
    video_ids = [c.video_id for c in result.chunks]
    assert video_ids == ["v1", "v2", "v3"]
    # Best distances per source
    assert result.chunks[0].distance == 0.05
    assert result.chunks[1].distance == 0.10
    assert result.chunks[2].distance == 0.12
    # candidates_evaluated reflects all chunks before aggregation
    assert result.candidates_evaluated == 6


def test_dynamic_pool_floor():
    from bibilab.pipeline.embed import _dynamic_pool

    assert _dynamic_pool(1) == 10  # 1*3=3, floored at 10
    assert _dynamic_pool(2) == 10  # 2*3=6, floored at 10
    assert _dynamic_pool(3) == 10  # 3*3=9, floored at 10


def test_dynamic_pool_scaling():
    from bibilab.pipeline.embed import _dynamic_pool

    assert _dynamic_pool(5) == 15
    assert _dynamic_pool(10) == 30


def test_dynamic_pool_ceiling():
    from bibilab.pipeline.embed import _dynamic_pool

    assert _dynamic_pool(20) == 60  # 20*3=60, exactly at ceiling
    assert _dynamic_pool(50) == 60  # capped
    assert _dynamic_pool(200) == 60  # capped


@pytest.mark.asyncio
async def test_retrieve_pool_at_least_top_k(tmp_bibilab_home):
    """effective_top_k must be >= params.top_k even when _dynamic_pool is smaller.
    E.g. analytical query on a 1-source list: pool=10 but top_k=12."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False))

    with patch(
        "bibilab.pipeline.embed.hybrid_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_hybrid:
        await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=4, top_k=12))

    mock_hybrid.assert_called_once_with("query", ["src1"], cfg, effective_top_k=12)


@pytest.mark.asyncio
async def test_retrieve_uses_candidate_pool(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import _dynamic_pool, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5))

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed.query_chunks", new_callable=AsyncMock) as mock_qc,
    ):
        mock_map.return_value = {"s1": "v1"}
        mock_qc.return_value = []

        await retrieve("test query", ["s1"], cfg, params=RetrievalParams(depth_per_source=1, top_k=10))

        mock_qc.assert_called_once_with("test query", ["s1"], cfg, top_k=_dynamic_pool(1), video_ids=["v1"])


# --- _adaptive_depth unit tests ---


def test_adaptive_depth_single_source_relaxes_to_top_k():
    from bibilab.pipeline.embed import _adaptive_depth

    # 1 source in pool: depth grows to top_k so single-source queries aren't
    # capped to 2 chunks (regression introduced by #287 scoped queries).
    assert _adaptive_depth(spec_depth=2, top_k=8, num_sources_in_pool=1) == 8


def test_adaptive_depth_few_sources_partial_relax():
    from bibilab.pipeline.embed import _adaptive_depth

    # 2 sources, top_k=8 → 4 each (ceil(8/2))
    assert _adaptive_depth(spec_depth=2, top_k=8, num_sources_in_pool=2) == 4


def test_adaptive_depth_many_sources_preserves_spec():
    from bibilab.pipeline.embed import _adaptive_depth

    # 8 sources, top_k=8 → spec wins (ceil(8/8)=1 < 2)
    assert _adaptive_depth(spec_depth=2, top_k=8, num_sources_in_pool=8) == 2


def test_adaptive_depth_zero_sources_returns_spec():
    from bibilab.pipeline.embed import _adaptive_depth

    # Empty pool: return spec to avoid divide-by-zero
    assert _adaptive_depth(spec_depth=2, top_k=8, num_sources_in_pool=0) == 2


@pytest.mark.asyncio
async def test_retrieve_depth_two_keeps_multiple_per_source(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import RetrievedChunk, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.5, reranking_enabled=False))

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

    def make_chunk(content, score_val):
        c = RetrievedChunk(
            content=content,
            video_title="Video 1",
            timestamp_start=0.0,
            timestamp_end=5.0,
            video_id="v1",
            distance=0.1,
        )
        c.score = score_val
        return c

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=[make_chunk("c1", 0.5), make_chunk("c2", 0.3), make_chunk("c3", 0.1)],
        ),
    ):
        mock_map.return_value = {"s1": "v1"}

        result = await retrieve("test query", ["s1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert len(result.chunks) == 3
    assert all(c.video_id == "v1" for c in result.chunks)


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
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import _dynamic_pool, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False))

    chunks = [_make_chunk(content="a", video_id="v1")]

    with patch(
        "bibilab.pipeline.embed.hybrid_search",
        new_callable=AsyncMock,
        return_value=chunks,
    ) as mock_hybrid:
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    mock_hybrid.assert_called_once_with("query", ["src1"], cfg, effective_top_k=_dynamic_pool(1))
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_retrieve_hybrid_disabled_skips_fts(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import _dynamic_pool, retrieve

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
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    mock_hybrid.assert_not_called()
    mock_vector.assert_called_once_with("query", ["src1"], cfg, top_k=_dynamic_pool(1))
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_retrieve_uses_candidate_pool_before_rerank(tmp_bibilab_home):
    """retrieve() must pull dynamic pool candidates and feed the full pool
    to the reranker, so all candidates contribute to source coverage. Then _diverse_top_k
    trims to params.top_k for the LLM input."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import _dynamic_pool, retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True))

    candidate_chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}") for i in range(10)]

    with (
        patch(
            "bibilab.pipeline.embed.hybrid_search",
            new_callable=AsyncMock,
            return_value=candidate_chunks,
        ) as mock_hybrid,
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=candidate_chunks,
        ) as mock_rerank,
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    mock_hybrid.assert_called_once_with("query", ["src1"], cfg, effective_top_k=_dynamic_pool(1))
    mock_rerank.assert_called_once_with("query", candidate_chunks, top_k=len(candidate_chunks))
    assert len(result.chunks) == 5
    assert result.candidates_evaluated == len(candidate_chunks)
    # sources_with_hits reflects result_chunks (what the LLM actually saw), not pool size
    assert result.sources_with_hits == 5


def test_rrf_fuse_ranks_doc_in_both_lists_above_doc_in_one():
    from bibilab.pipeline.embed import _rrf_fuse

    doc_a = _make_chunk(content="a", video_id="v1", distance=0.1)
    doc_b = _make_chunk(content="b", video_id="v2", distance=0.2)
    doc_c = _make_chunk(content="c", video_id="v3", distance=0.3)

    vec_list = [doc_a, doc_b, doc_c]
    fts_list = [doc_a, doc_c]

    result = _rrf_fuse(vec_list, fts_list)

    assert result[0].content == "a"
    assert result.index(doc_c) < result.index(doc_b)


@pytest.mark.asyncio
async def test_diverse_top_k_keeps_highest_scoring_per_source(tmp_bibilab_home):
    """With depth=1 and score-descending input, _diverse_top_k keeps the best chunk per source."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
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

    # In production, reranker returns chunks in score-descending order
    chunks = [v1_high, v2_high, v2_low, v1_low]

    with patch(
        "bibilab.pipeline.embed.hybrid_search",
        new_callable=AsyncMock,
        return_value=chunks,
    ):
        result = await retrieve("query", ["s1", "s2"], cfg, params=RetrievalParams(depth_per_source=1, top_k=2))

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
    from bibilab.models._enums import RetrievalParams
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

        result = await retrieve(
            "test query", ["s1", "s2", "s3"], cfg, params=RetrievalParams(depth_per_source=2, top_k=2)
        )

    # candidates_evaluated should reflect the 5 chunks retrieved BEFORE reranking trimmed to top_k=2
    assert result.candidates_evaluated == 5
    # But chunks returned should be the top_k after reranking
    assert len(result.chunks) == 2


# --- _diverse_top_k unit tests ---


def test_diverse_top_k_depth_one_one_per_source():
    from bibilab.pipeline.embed import _diverse_top_k

    chunks = [
        _make_chunk(content="a1", video_id="v1"),
        _make_chunk(content="a2", video_id="v1"),
        _make_chunk(content="b1", video_id="v2"),
        _make_chunk(content="b2", video_id="v2"),
        _make_chunk(content="c1", video_id="v3"),
        _make_chunk(content="c2", video_id="v3"),
    ]
    result = _diverse_top_k(chunks, depth=1, k=3)
    assert len(result) == 3
    assert {c.video_id for c in result} == {"v1", "v2", "v3"}


def test_diverse_top_k_depth_two_allows_two_per_source():
    from bibilab.pipeline.embed import _diverse_top_k

    chunks = [
        _make_chunk(content="a1", video_id="v1"),
        _make_chunk(content="a2", video_id="v1"),
        _make_chunk(content="a3", video_id="v1"),
        _make_chunk(content="b1", video_id="v2"),
        _make_chunk(content="b2", video_id="v2"),
        _make_chunk(content="c1", video_id="v3"),
        _make_chunk(content="c2", video_id="v3"),
    ]
    result = _diverse_top_k(chunks, depth=2, k=4)
    # depth=2 allows ≤2 per source; k=4 is filled before leftovers
    counts: dict[str, int] = {}
    for c in result:
        counts[c.video_id] = counts.get(c.video_id, 0) + 1
    assert all(v <= 2 for v in counts.values())


def test_diverse_top_k_strict_cap_single_source():
    from bibilab.pipeline.embed import _diverse_top_k

    chunks = [_make_chunk(content=f"c{i}", video_id="v1") for i in range(10)]
    # depth=2 strictly caps single-source returns to 2, regardless of k
    result = _diverse_top_k(chunks, depth=2, k=5)
    assert len(result) == 2
    assert all(c.video_id == "v1" for c in result)


def test_diverse_top_k_short_return_when_cap_blocks_fill():
    from bibilab.pipeline.embed import _diverse_top_k

    chunks = [
        _make_chunk(content="a1", video_id="v1"),
        _make_chunk(content="a2", video_id="v1"),
        _make_chunk(content="b1", video_id="v2"),
        _make_chunk(content="b2", video_id="v2"),
    ]
    # depth=1, k=3: picks a1, b1 → cap blocks remaining slot, no leftover fill
    result = _diverse_top_k(chunks, depth=1, k=3)
    assert len(result) == 2
    assert {c.video_id for c in result} == {"v1", "v2"}


def test_retrieval_result_has_telemetry_fields():
    """Smoke test: new I-4 telemetry fields exist with safe defaults."""
    from bibilab.pipeline.embed import RetrievalResult

    r = RetrievalResult(
        chunks=[],
        candidates_evaluated=0,
        sources_with_hits=0,
        sources_total=0,
        source_coverage=[],
    )
    assert r.dropped_by_gate == 0
    assert r.reranked is False


# --- _quantile_gate unit tests ---


def _chunk_with_score(score: float, vid: str = "v1"):
    from bibilab.pipeline.embed import RetrievedChunk

    return RetrievedChunk(
        content="c",
        video_title="t",
        timestamp_start=0.0,
        timestamp_end=1.0,
        video_id=vid,
        distance=0.0,
        score=score,
    )


def test_quantile_gate_drops_below_margin():
    from bibilab.pipeline.embed import _quantile_gate

    chunks = [_chunk_with_score(s) for s in (8.0, 7.5, 3.2, -1.0)]
    # top=8, median=5.35, threshold = max(5.35, 8-4) = 5.35 → keep [8.0, 7.5]
    kept = _quantile_gate(chunks)
    scores = [c.score for c in kept]
    assert scores == [8.0, 7.5]


def test_quantile_gate_single_top_honors_min_keep():
    from bibilab.pipeline.embed import _quantile_gate

    chunks = [_chunk_with_score(5.0)]
    kept = _quantile_gate(chunks)
    assert len(kept) == 1
    assert kept[0].score == 5.0


def test_quantile_gate_empty_input_returns_empty():
    from bibilab.pipeline.embed import _quantile_gate

    assert _quantile_gate([]) == []


def test_quantile_gate_all_within_margin_keeps_all():
    from bibilab.pipeline.embed import _quantile_gate

    chunks = [_chunk_with_score(s) for s in (8.0, 7.8, 7.5, 7.2)]
    # top=8, median=7.5, threshold = max(7.5, 4.0) = 7.5 → keep [8.0, 7.8, 7.5]
    kept = _quantile_gate(chunks)
    assert [c.score for c in kept] == [8.0, 7.8, 7.5]


# ─── Task 7: reranked flag wiring ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_reranked_flag_true_on_success(monkeypatch, tmp_bibilab_home):
    """When rerank succeeds, reranked=True and dropped_by_gate reflects gate."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline import embed as embed_mod

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [[f"chunk{i} content" for i in range(6)]],
        "metadatas": [
            [
                {
                    "video_id": "v1",
                    "video_title": "V",
                    "timestamp_start": float(i) * 10,
                    "timestamp_end": float(i) * 10 + 9.9,
                }
                for i in range(6)
            ]
        ],
        "distances": [[0.1] * 6],
    }

    async def fake_rerank(query, chunks, top_k):
        scores = [8.0, 7.5, 3.2, -1.0, -2.0, -3.0][: len(chunks)]
        for c, s in zip(chunks, scores, strict=False):
            c.score = s
        return chunks

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        patch("bibilab.pipeline.rerank.rerank", side_effect=fake_rerank),
    ):
        mock_map.return_value = {"source-1": "v1"}

        result = await embed_mod.retrieve(
            query_text="q",
            source_ids=["source-1"],
            cfg=cfg,
            params=RetrievalParams(depth_per_source=2, top_k=8),
            scoped_source_ids=None,
        )
        assert result.reranked is True
        # Gate keeps [8.0, 7.5] → drops 4
        assert result.dropped_by_gate == 4
        assert len(result.chunks) == 2


@pytest.mark.asyncio
async def test_retrieve_reranked_flag_false_when_disabled(tmp_bibilab_home):
    """reranking_enabled=False → reranked=False, dropped_by_gate=0."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline import embed as embed_mod

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False, hybrid_enabled=False))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["only one"]],
        "metadatas": [[{"video_id": "v1", "video_title": "V", "timestamp_start": 0.0, "timestamp_end": 10.0}]],
        "distances": [[0.1]],
    }

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"source-1": "v1"}

        result = await embed_mod.retrieve(
            query_text="q",
            source_ids=["source-1"],
            cfg=cfg,
            params=RetrievalParams(depth_per_source=2, top_k=8),
            scoped_source_ids=None,
        )
        assert result.reranked is False
        assert result.dropped_by_gate == 0


@pytest.mark.asyncio
async def test_retrieve_reranked_flag_false_on_exception(monkeypatch, tmp_bibilab_home):
    """Rerank raises → reranked=False, gate bypassed."""
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline import embed as embed_mod

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["one", "two"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "V", "timestamp_start": 0.0, "timestamp_end": 10.0},
                {"video_id": "v1", "video_title": "V", "timestamp_start": 10.0, "timestamp_end": 20.0},
            ]
        ],
        "distances": [[0.1, 0.2]],
    }

    async def boom(*a, **kw):
        raise RuntimeError("model missing")

    with (
        patch("bibilab.pipeline.embed.get_video_ids_for_sources", new_callable=AsyncMock) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        patch("bibilab.pipeline.rerank.rerank", side_effect=boom),
    ):
        mock_map.return_value = {"source-1": "v1"}

        result = await embed_mod.retrieve(
            query_text="q",
            source_ids=["source-1"],
            cfg=cfg,
            params=RetrievalParams(depth_per_source=2, top_k=8),
            scoped_source_ids=None,
        )
        assert result.reranked is False
        assert result.dropped_by_gate == 0
