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
        result = await retrieve("query", ["src1"], cfg, top_k=5)

    mock_rerank.assert_called_once_with("query", chunks, top_k=5)
    assert result.chunks == list(reversed(chunks))


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_rerank_raises(tmp_bibilab_home, caplog):
    import logging

    from bibilab.config import BibilabConfig, RagConfig
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
        result = await retrieve("query", ["src1"], cfg, top_k=3)

    assert len(result.chunks) == 3
    assert [c.content for c in result.chunks] == ["c0", "c1", "c2"]
    assert any("Reranking failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_retrieve_with_reranking_disabled(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
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
        result = await retrieve("query", ["src1"], cfg, top_k=5)

    mock_rerank.assert_not_called()
    assert result.chunks == chunks


@pytest.mark.asyncio
async def test_reranker_lazy_singleton():
    import bibilab.pipeline.rerank as rerank_mod

    original = rerank_mod._reranker
    try:
        rerank_mod._reranker = None

        mock_ce_class = MagicMock()
        mock_ce_instance = MagicMock()
        mock_ce_class.return_value = mock_ce_instance

        with patch("bibilab.pipeline.rerank.CrossEncoder", mock_ce_class, create=True):
            # Patch the import inside _get_reranker
            with patch.dict("sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_class)}):
                result1 = rerank_mod._get_reranker()
                result2 = rerank_mod._get_reranker()

        assert result1 is result2
        mock_ce_class.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
    finally:
        rerank_mod._reranker = original
