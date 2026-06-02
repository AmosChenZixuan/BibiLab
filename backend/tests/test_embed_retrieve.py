"""Retrieve() behaviour after the v2 gate/diversity/neighbor deletion (#370)."""

from __future__ import annotations

import pytest

from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
from bibilab.pipeline import embed
from bibilab.pipeline.embed import RetrievedChunk, retrieve


def _cfg(*, reranking: bool = True) -> BibilabConfig:
    return BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(reranking_enabled=reranking, hybrid_enabled=True, max_distance=10.0),
    )


@pytest.mark.asyncio
async def test_retrieve_returns_topk_by_rerank_no_gate(monkeypatch):
    """No gate: retrieve returns exactly top_k chunks in rerank order, even when
    lower-ranked chunks would have been gated out in v1."""
    pool = [
        RetrievedChunk(
            content=f"c{i}",
            video_title="t",
            timestamp_start=float(i),
            timestamp_end=float(i) + 1,
            source_id="s1",
            distance=0.0,
            score=10.0 - i,
            sequence_index=i,
        )
        for i in range(20)
    ]

    async def fake_hybrid(*a, **k):  # noqa: ANN001
        return list(pool)

    async def fake_rerank(query, chunks, top_k):  # noqa: ANN001
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    monkeypatch.setattr(embed, "hybrid_search", fake_hybrid)
    monkeypatch.setattr("bibilab.pipeline.rerank.rerank", fake_rerank)

    result = await retrieve("q", ["s1"], _cfg(), top_k=8)

    assert len(result.chunks) == 8
    assert [c.content for c in result.chunks] == [f"c{i}" for i in range(8)]
    assert result.reranked is True


@pytest.mark.asyncio
async def test_retrieve_keeps_multiple_sources_no_diversity_cap(monkeypatch):
    """No per-source diversity cap: a single loud source may fill many of the
    top_k slots; recall is left to the rerank order."""
    pool = [
        RetrievedChunk(
            content=f"a{i}",
            video_title="ta",
            timestamp_start=float(i),
            timestamp_end=float(i) + 1,
            source_id="sa",
            distance=0.0,
            score=100.0 - i,
            sequence_index=i,
        )
        for i in range(6)
    ] + [
        RetrievedChunk(
            content="b0",
            video_title="tb",
            timestamp_start=0.0,
            timestamp_end=1.0,
            source_id="sb",
            distance=0.0,
            score=1.0,
            sequence_index=0,
        )
    ]

    async def fake_hybrid(*a, **k):  # noqa: ANN001
        return list(pool)

    async def fake_rerank(query, chunks, top_k):  # noqa: ANN001
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    monkeypatch.setattr(embed, "hybrid_search", fake_hybrid)
    monkeypatch.setattr("bibilab.pipeline.rerank.rerank", fake_rerank)

    result = await retrieve("q", ["sa", "sb"], _cfg(), top_k=8)

    # No diversity cap → all 6 'sa' chunks present (v1 would cap at depth=2).
    assert sum(c.source_id == "sa" for c in result.chunks) == 6
    assert any(c.source_id == "sb" for c in result.chunks)


@pytest.mark.asyncio
async def test_retrieve_empty_pool_returns_empty(monkeypatch):
    async def fake_hybrid(*a, **k):  # noqa: ANN001
        return []

    monkeypatch.setattr(embed, "hybrid_search", fake_hybrid)
    result = await retrieve("q", ["s1"], _cfg(), top_k=8)
    assert result.chunks == []
    assert result.sources_with_hits == 0
