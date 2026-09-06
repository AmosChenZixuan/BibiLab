"""#728: retrieve() pre-LLM candidate pool scales by section count, not source count.

Pure unit tests — they mock `count_sections_for_sources` and the
hybrid/rerank search pipeline so the section-pool formula can be pinned
without spinning up real Chroma or SQLite.
"""

from __future__ import annotations

import pytest

from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
from bibilab.pipeline import embed
from bibilab.pipeline.embed import retrieve


def _cfg() -> BibilabConfig:
    return BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(reranking_enabled=False, hybrid_enabled=True, max_distance=10.0),
    )


def _install_hybrid_capture(monkeypatch, captured: list[int]) -> None:
    async def fake_hybrid(*a, **k):  # noqa: ANN001
        captured.append(k.get("effective_top_k"))
        return []

    monkeypatch.setattr(embed, "hybrid_search", fake_hybrid)


@pytest.mark.asyncio
async def test_pool_scales_by_section_count_not_source(monkeypatch):
    """#728 AC5: 4 sources / 18 sections → effective_top_k=54.

    The source-based formula yields 12 on the same inputs (4*3, floored
    to 10 then to 12). This test must fail against the pre-fix code.
    """
    captured: list[int] = []

    async def fake_count(source_ids: list[str]) -> int:
        return 18  # 诡秘之主 shape: 4 sources, 5–6 sections apiece

    monkeypatch.setattr(embed, "count_sections_for_sources", fake_count)
    _install_hybrid_capture(monkeypatch, captured)

    await retrieve("q", ["s1", "s2", "s3", "s4"], _cfg(), top_k=8)

    assert captured[0] == 54  # min(max(18*3, 8, 10), 60)


@pytest.mark.asyncio
async def test_pool_follows_search_pool_when_facet_scoped(monkeypatch):
    """#728 AC2: facet scoping shrinks the pool to the scoped section count."""
    captured: list[int] = []

    async def fake_count(source_ids: list[str]) -> int:
        # search_pool carries the facet-narrowed set; full pool is wider.
        return 5 if len(source_ids) <= 1 else 18

    monkeypatch.setattr(embed, "count_sections_for_sources", fake_count)
    _install_hybrid_capture(monkeypatch, captured)

    await retrieve(
        "q",
        ["s1", "s2", "s3", "s4"],
        _cfg(),
        top_k=8,
        scoped_source_ids=["s1"],
    )

    assert captured[0] == 15  # min(max(5*3, 8, 10), 60)


@pytest.mark.asyncio
async def test_pool_caps_at_60(monkeypatch):
    """#728 AC4 (60 cap): M*3 > 60 → effective_top_k == 60."""
    captured: list[int] = []

    async def fake_count(source_ids: list[str]) -> int:
        return 50  # 50*3 = 150, capped to 60

    monkeypatch.setattr(embed, "count_sections_for_sources", fake_count)
    _install_hybrid_capture(monkeypatch, captured)

    await retrieve("q", ["s1", "s2"], _cfg(), top_k=8)

    assert captured[0] == 60


@pytest.mark.asyncio
async def test_pool_floors_at_top_k_when_top_k_wins(monkeypatch):
    """#728 AC4 (top_k floor): top_k > sections*3 → effective_top_k == top_k."""
    captured: list[int] = []

    async def fake_count(source_ids: list[str]) -> int:
        return 3  # 3*3 = 9 < top_k=20

    monkeypatch.setattr(embed, "count_sections_for_sources", fake_count)
    _install_hybrid_capture(monkeypatch, captured)

    await retrieve("q", ["s1"], _cfg(), top_k=20)

    assert captured[0] == 20
