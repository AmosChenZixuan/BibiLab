"""Intra-turn chunk dedup: pure partition helper + execute_find_passages wiring."""

import logging
from types import SimpleNamespace

import pytest

from bibilab.config import BibilabConfig
from bibilab.pipeline import chat_tools
from bibilab.pipeline.chat_tools import _partition_unseen_chunks


def _chunk(source_id: str, start: float, end: float):
    # Mirrors the fields execute_find_passages reads off a retrieve chunk.
    return SimpleNamespace(
        source_id=source_id,
        timestamp_start=start,
        timestamp_end=end,
        content=f"{source_id}@{int(start)}",
    )


def test_partition_keeps_first_occurrence_and_records_id():
    seen: set[str] = set()
    chunks = [_chunk("s1", 10, 20), _chunk("s1", 30, 40)]
    new = _partition_unseen_chunks(chunks, seen)
    assert [c.content for c in new] == ["s1@10", "s1@30"]
    assert seen == {"s1_10_20", "s1_30_40"}


def test_partition_drops_ids_already_seen():
    seen = {"s1_10_20"}
    chunks = [_chunk("s1", 10, 20), _chunk("s2", 5, 9)]
    new = _partition_unseen_chunks(chunks, seen)
    assert [c.source_id for c in new] == ["s2"]
    assert seen == {"s1_10_20", "s2_5_9"}


def test_partition_dedups_within_a_single_call():
    seen: set[str] = set()
    chunks = [_chunk("s1", 10, 20), _chunk("s1", 10, 20)]
    new = _partition_unseen_chunks(chunks, seen)
    assert len(new) == 1


def test_partition_truncates_subsecond_timestamps_to_int():
    # chunk_id uses int(ts) so 20.4 and 20.9 collide with 20 — matches the
    # existing chunk_id format at chat_tools.py:456/478.
    seen: set[str] = set()
    _partition_unseen_chunks([_chunk("s1", 10.4, 20.9)], seen)
    assert seen == {"s1_10_20"}


def _retrieve_chunk(source_id, start, end, title="T", score=1.0):
    return SimpleNamespace(
        source_id=source_id,
        content=f"{source_id} body @{int(start)}",
        video_title=title,
        timestamp_start=start,
        timestamp_end=end,
        score=score,
        seg_start=None,  # None → speaker reconstruction skipped, no DB hit
        seg_end=None,
    )


def _retrieve_result(chunks):
    sources = {c.source_id for c in chunks}
    return SimpleNamespace(
        chunks=chunks,
        source_coverage=[SimpleNamespace(source_id=s, video_title="T") for s in sources],
        candidates_evaluated=len(chunks),
        sources_with_hits=len(sources),
        sources_total=len(sources),
        reranked=True,
    )


def _cfg():
    return BibilabConfig()  # retrieve is mocked, so cfg content is irrelevant


@pytest.mark.asyncio
async def test_find_passages_suppresses_already_seen_chunk(monkeypatch):
    calls = [
        _retrieve_result([_retrieve_chunk("s1", 10, 20), _retrieve_chunk("s1", 30, 40)]),
        _retrieve_result([_retrieve_chunk("s1", 10, 20)]),  # overlaps call 1
    ]

    async def fake_retrieve(**kwargs):
        return calls.pop(0)

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    seen: set[str] = set()
    registry: dict = {}

    r1 = await chat_tools.execute_find_passages(
        query="q1", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )
    assert "s1 body @10" in r1["_chunks"]
    assert "s1 body @30" in r1["_chunks"]

    r2 = await chat_tools.execute_find_passages(
        query="q2", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )
    # The only hit was already shown in call 1 → collapse to a fact, no chunk body.
    assert "already retrieved earlier this turn" in r2["_chunks"]
    assert "s1 body @10" not in r2["_chunks"]


@pytest.mark.asyncio
async def test_find_passages_no_seen_set_is_unchanged(monkeypatch):
    async def fake_retrieve(**kwargs):
        return _retrieve_result([_retrieve_chunk("s1", 10, 20)])

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    r = await chat_tools.execute_find_passages(query="q", source_ids=["s1"], cfg=_cfg(), registry={})
    assert "s1 body @10" in r["_chunks"]


@pytest.mark.asyncio
async def test_find_passages_partial_overlap_preserves_source_index(monkeypatch):
    """Production scenario: call 2 has partial overlap with call 1. The
    already-seen chunk is dropped, the new chunk is rendered, and the source's
    [N] is preserved (it was registered in call 1)."""
    calls = [
        _retrieve_result([_retrieve_chunk("s1", 10, 20), _retrieve_chunk("s1", 30, 40)]),
        _retrieve_result([_retrieve_chunk("s1", 10, 20), _retrieve_chunk("s1", 50, 60)]),
    ]

    async def fake_retrieve(**kwargs):
        return calls.pop(0)

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    seen: set[str] = set()
    registry: dict = {}

    r1 = await chat_tools.execute_find_passages(
        query="q1", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )
    r2 = await chat_tools.execute_find_passages(
        query="q2", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )

    # r1 baseline: both chunks rendered, no dedup.
    assert "s1 body @10" in r1["_chunks"]
    assert "s1 body @30" in r1["_chunks"]
    # s1's [N] was assigned in r1 and must survive in r2.
    assert registry["s1"].index == 1
    # s1@50 is new — rendered. s1@10 was already shown — suppressed.
    assert "s1 body @50" in r2["_chunks"]
    assert "s1 body @10" not in r2["_chunks"]
    # All-shown fact must NOT fire (partial overlap).
    assert "already retrieved earlier this turn" not in r2["_chunks"]
    # _raw_chunks mirrors new_chunks (only the new one).
    assert len(r2["_raw_chunks"]) == 1
    assert r2["_raw_chunks"][0]["timestamp_start"] == 50


@pytest.mark.asyncio
async def test_find_passages_different_sources_all_unseen(monkeypatch):
    """Canary: dedup key is chunk-level, not source-level. Different sources
    across calls must NOT be considered duplicates."""
    calls = [
        _retrieve_result([_retrieve_chunk("s1", 10, 20)]),
        _retrieve_result([_retrieve_chunk("s2", 30, 40)]),
    ]

    async def fake_retrieve(**kwargs):
        return calls.pop(0)

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    seen: set[str] = set()
    registry: dict = {}

    r1 = await chat_tools.execute_find_passages(
        query="q1", source_ids=["s1", "s2"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )
    r2 = await chat_tools.execute_find_passages(
        query="q2", source_ids=["s1", "s2"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
    )

    assert "s1 body @10" in r1["_chunks"]
    assert "s2 body @30" in r2["_chunks"]  # would fail if dedup key were source_id only
    assert "already retrieved earlier this turn" not in r2["_chunks"]


@pytest.mark.asyncio
async def test_execute_tool_forwards_seen_chunk_ids_to_find_passages(monkeypatch):
    """The kwarg-forwarding seam between execute_tool and execute_find_passages.
    A typo in the kwargs dict at chat_tools.py:583 would orphan dedup state —
    this test pins the contract."""
    seen_at_fp: list = []

    real_fp = chat_tools.execute_find_passages

    async def spy_fp(*args, **kwargs):
        seen_at_fp.append(kwargs.get("seen_chunk_ids"))
        return await real_fp(*args, **kwargs)

    monkeypatch.setattr(chat_tools, "execute_find_passages", spy_fp)

    async def fake_retrieve(**kwargs):
        return _retrieve_result([_retrieve_chunk("s1", 10, 20)])

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    seen: set[str] = set()
    await chat_tools.execute_tool(
        tool_name="find_passages",
        arguments={"query": "q"},
        source_ids=["s1"],
        cfg=_cfg(),
        seen_chunk_ids=seen,
    )

    assert len(seen_at_fp) == 1
    assert seen_at_fp[0] is seen  # same object forwarded, not a copy


@pytest.mark.asyncio
async def test_find_passages_logs_dedup_count(monkeypatch, caplog):
    """find_passages_intraturn_dedup is the load-bearing measurement gating the
    cross-turn retention POC. Pin its firing so a future refactor can't quietly
    lose the observability."""
    calls = [
        _retrieve_result([_retrieve_chunk("s1", 10, 20), _retrieve_chunk("s1", 30, 40)]),
        _retrieve_result([_retrieve_chunk("s1", 10, 20)]),
    ]

    async def fake_retrieve(**kwargs):
        return calls.pop(0)

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    seen: set[str] = set()
    registry: dict = {}

    with caplog.at_level(logging.INFO, logger="bibilab.pipeline.chat_tools"):
        await chat_tools.execute_find_passages(
            query="q1", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
        )
        # Second call overlaps with first by 1 chunk → log fires with deduped=1.
        await chat_tools.execute_find_passages(
            query="q2", source_ids=["s1"], cfg=_cfg(), registry=registry, seen_chunk_ids=seen
        )

    matches = [
        rec for rec in caplog.records if "find_passages_intraturn_dedup" in rec.message and "deduped=1" in rec.message
    ]
    assert matches, f"expected at least one dedup log; got {[r.message for r in caplog.records]}"
