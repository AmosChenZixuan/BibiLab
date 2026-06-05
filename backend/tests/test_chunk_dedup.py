"""Intra-turn chunk dedup: pure partition helper + execute_find_passages wiring."""

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
    # existing chunk_id format at chat_tools.py:419/441.
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
