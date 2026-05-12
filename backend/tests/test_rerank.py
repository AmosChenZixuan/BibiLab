from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_chunk(content="chunk", video_title="vid", ts_start=0.0, ts_end=1.0, video_id="v1", distance=0.5):
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
async def test_rerank_sorts_by_cross_encoder_score():
    from bibilab.pipeline.rerank import rerank

    chunks = [_make_chunk(content="low"), _make_chunk(content="high"), _make_chunk(content="mid")]

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.1, 0.9, 0.5]

    with patch("bibilab.pipeline.rerank._get_reranker", return_value=mock_reranker):
        result = await rerank("query", chunks, top_k=3)

    assert [c.content for c in result] == ["high", "mid", "low"]
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.5)
    assert result[2].score == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_rerank_returns_top_k():
    from bibilab.pipeline.rerank import rerank

    chunks = [_make_chunk(content=f"c{i}") for i in range(5)]

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]

    with patch("bibilab.pipeline.rerank._get_reranker", return_value=mock_reranker):
        result = await rerank("query", chunks, top_k=2)

    assert len(result) == 2
    assert result[0].content == "c1"
    assert result[1].content == "c4"


@pytest.mark.asyncio
async def test_rerank_empty_chunks():
    from bibilab.pipeline.rerank import rerank

    result = await rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_rerank_single_chunk():
    from bibilab.pipeline.rerank import rerank

    chunk = _make_chunk(content="only")
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.8]

    with patch("bibilab.pipeline.rerank._get_reranker", return_value=mock_reranker):
        result = await rerank("query", [chunk], top_k=5)

    assert len(result) == 1
    assert result[0].content == "only"
    assert result[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_retrieve_with_reranking_enabled(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False))

    chunks = [_make_chunk(content="a", distance=0.2), _make_chunk(content="b", distance=0.1)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=list(reversed(chunks)),
        ) as mock_rerank,
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    mock_rerank.assert_called_once_with("query", chunks, top_k=len(chunks))
    assert result.chunks == list(reversed(chunks))


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_rerank_raises(tmp_bibilab_home, caplog):
    import logging

    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False))

    chunks = [_make_chunk(content=f"c{i}", distance=0.1 * i) for i in range(5)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            side_effect=RuntimeError("model download failed"),
        ),
        caplog.at_level(logging.WARNING, logger="bibilab.pipeline.embed"),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=3))

    assert len(result.chunks) == 3
    assert [c.content for c in result.chunks] == ["c0", "c1", "c2"]
    assert any("Reranking failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_retrieve_with_reranking_disabled(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(max_distance=1.0, reranking_enabled=False, hybrid_enabled=False))

    chunks = [_make_chunk(content="a", distance=0.2)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
        ) as mock_rerank,
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    mock_rerank.assert_not_called()
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_broad_mode_covers_candidate_pool_sources(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.0)
    )

    chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}", video_title=f"vid{i}") for i in range(8)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=1, top_k=8))

    assert len(result.source_coverage) == 8
    assert len(result.chunks) == 8
    assert len({c.video_id for c in result.chunks}) == 8


@pytest.mark.asyncio
async def test_focused_mode_unchanged(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.0)
    )

    chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}", video_title=f"vid{i}") for i in range(8)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=chunks[:5],
        ),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert len(result.chunks) == 5


@pytest.mark.asyncio
async def test_rerank_floor_drops_low_scores(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.0)
    )

    chunks = [_make_chunk(content=f"c{i}") for i in range(5)]
    reranked = [_make_chunk(content=f"c{i}", distance=0.0) for i in range(5)]
    for i, c in enumerate(reranked):
        c.score = [-2, -1, 0.5, 2, 5][i]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=reranked,
        ),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert len(result.chunks) == 2
    assert [c.content for c in result.chunks] == ["c3", "c4"]


@pytest.mark.asyncio
async def test_rerank_floor_disabled_when_none(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=None)
    )

    chunks = [_make_chunk(content=f"c{i}") for i in range(5)]
    reranked = [_make_chunk(content=f"c{i}", distance=0.0) for i in range(5)]
    for i, c in enumerate(reranked):
        c.score = [-2, -1, 0.5, 2, 5][i]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=reranked,
        ),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    # _quantile_gate: top=5, median=-1, threshold=max(-1, 5-4)=1 → keeps [c3,c4]
    # depth=2 from 1 source → 2 chunks
    assert len(result.chunks) == 2
    assert [c.content for c in result.chunks] == ["c3", "c4"]


@pytest.mark.asyncio
async def test_rerank_failure_skips_floor(tmp_bibilab_home, caplog):
    import logging

    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.5)
    )

    chunks = [_make_chunk(content=f"c{i}", distance=0.1 * i) for i in range(5)]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            side_effect=RuntimeError("model download failed"),
        ),
        caplog.at_level(logging.WARNING, logger="bibilab.pipeline.embed"),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=3))

    assert len(result.chunks) == 3
    assert [c.content for c in result.chunks] == ["c0", "c1", "c2"]
    assert any("Reranking failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_broad_mode_respects_floor(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(rag=RagConfig(reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.0))

    chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}") for i in range(4)]

    reranked = list(chunks)
    for c, s in zip(reranked, [-1.0, -0.5, 0.5, 1.0]):
        c.score = s

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch(
            "bibilab.pipeline.rerank.rerank",
            new_callable=AsyncMock,
            return_value=reranked,
        ),
    ):
        result = await retrieve("q", ["src1"], cfg, params=RetrievalParams(depth_per_source=1, top_k=4))

    assert {c.video_id for c in result.chunks} == {"v1", "v2", "v3"}
    # sources_with_hits now reflects result_chunks (what the LLM actually saw), not pool size
    assert result.sources_with_hits == 3


@pytest.mark.asyncio
async def test_sources_with_hits_reflects_result_chunks(tmp_bibilab_home):
    """sources_with_hits reflects the distinct video_ids in result_chunks (post diverse-top-k), not the rerank pool.

    Regression for #9 — ObsChip should show real retrieval coverage, not the LLM-input slice.
    """
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.models._enums import RetrievalParams
    from bibilab.pipeline.embed import retrieve

    cfg = BibilabConfig(
        rag=RagConfig(max_distance=1.0, reranking_enabled=True, hybrid_enabled=False, rerank_min_score=0.0)
    )

    chunks = [_make_chunk(content=f"c{i}", video_id=f"v{i}", video_title=f"vid{i}") for i in range(8)]

    async def mock_rerank(query, chunks_arg, top_k):
        for c in chunks_arg:
            c.score = 1.0
        return list(chunks_arg)[:top_k]

    with (
        patch(
            "bibilab.pipeline.embed.query_chunks",
            new_callable=AsyncMock,
            return_value=chunks,
        ),
        patch("bibilab.pipeline.rerank.rerank", side_effect=mock_rerank),
    ):
        result = await retrieve("query", ["src1"], cfg, params=RetrievalParams(depth_per_source=2, top_k=5))

    assert result.sources_with_hits == 5
    assert len(result.chunks) == 5


@pytest.mark.asyncio
async def test_reranker_lazy_singleton():
    import bibilab.pipeline.rerank as rerank_mod

    original = rerank_mod._reranker
    try:
        rerank_mod._reranker = None

        mock_ce_instance = MagicMock()
        mock_ce_class = MagicMock(return_value=mock_ce_instance)

        with patch("bibilab.pipeline.rerank.ONNXCrossEncoder", mock_ce_class):
            result1 = rerank_mod._get_reranker()
            result2 = rerank_mod._get_reranker()

            assert result1 is result2
            mock_ce_class.assert_called_once()
    finally:
        rerank_mod._reranker = original


def test_get_collection_keys_by_name(tmp_bibilab_home):
    """_get_collection returns different collections for different config names.

    Regression test for #223: the module-level singleton was stored as a single
    reference (_chroma_collection = None) that never updated when
    transcript_collection_name changed. After the fix, _chroma_collections is
    a dict keyed by collection name, so different configs get different collections.
    """
    import bibilab.pipeline.embed as embed_mod
    from bibilab.config import BibilabConfig

    # Reset module-level state
    embed_mod._chroma_collections = {}

    collections_created = []

    class FakeCollection:
        def __init__(self, name):
            self.name = name

    def fake_get_or_create(name, embedding_function=None):
        coll = FakeCollection(name)
        collections_created.append(coll)
        return coll

    fake_client = MagicMock()
    fake_client.get_or_create_collection = fake_get_or_create

    # chromadb is imported lazily inside _get_collection, so patch the
    # top-level chromadb module so the local import picks up the mock.
    with patch("chromadb.PersistentClient", return_value=fake_client):
        cfg1 = BibilabConfig(transcript_collection_name="collection_a")
        cfg2 = BibilabConfig(transcript_collection_name="collection_b")

        coll_a = embed_mod._get_collection(cfg1)
        coll_b = embed_mod._get_collection(cfg2)

        # Two different configs must produce distinct collections
        assert coll_a is not coll_b
        assert coll_a.name == "collection_a"
        assert coll_b.name == "collection_b"

        # Calling again with the same name returns the cached collection
        coll_a2 = embed_mod._get_collection(cfg1)
        assert coll_a2 is coll_a

        assert len(collections_created) == 2

    # Clean up
    embed_mod._chroma_collections = {}
