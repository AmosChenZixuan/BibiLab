"""read_source tool: single-match resolution + narrative + citation binding (#371)."""

from __future__ import annotations

import pytest

from bibilab.pipeline import chat_tools
from bibilab.pipeline.chat_tools import (
    CitationRegistryEntry,
    _resolve_single_source,
    execute_read_source,
)


def _facets(mapping):  # helper to stub get_source_facets
    async def _f(source_ids):  # noqa: ANN001
        return {sid: mapping[sid] for sid in source_ids if sid in mapping}

    return _f


@pytest.mark.asyncio
async def test_resolve_by_source_id_in_pool():
    rid, err = await _resolve_single_source(["a", "b"], source_id="a", sequence_number=None, season_number=None)
    assert rid == "a" and err is None


@pytest.mark.asyncio
async def test_resolve_by_source_id_outside_pool_errors():
    rid, err = await _resolve_single_source(["a"], source_id="zzz", sequence_number=None, season_number=None)
    assert rid is None and "not in this list" in err


@pytest.mark.asyncio
async def test_resolve_by_facet_single_match(monkeypatch):
    monkeypatch.setattr(
        chat_tools,
        "get_source_facets",
        _facets({"a": {"sequence_number": 2, "season_number": 1}, "b": {"sequence_number": 3, "season_number": 1}}),
    )
    rid, err = await _resolve_single_source(["a", "b"], source_id=None, sequence_number=2, season_number=None)
    assert rid == "a" and err is None


@pytest.mark.asyncio
async def test_resolve_by_facet_no_match_errors(monkeypatch):
    monkeypatch.setattr(chat_tools, "get_source_facets", _facets({"a": {"sequence_number": 2, "season_number": 1}}))
    rid, err = await _resolve_single_source(["a"], source_id=None, sequence_number=9, season_number=None)
    assert rid is None and "no source" in err.lower()


@pytest.mark.asyncio
async def test_resolve_by_facet_multi_match_errors(monkeypatch):
    monkeypatch.setattr(
        chat_tools,
        "get_source_facets",
        _facets({"a": {"sequence_number": 2, "season_number": 1}, "b": {"sequence_number": 2, "season_number": 2}}),
    )
    rid, err = await _resolve_single_source(["a", "b"], source_id=None, sequence_number=2, season_number=None)
    assert rid is None and "ambiguous" in err.lower()


@pytest.mark.asyncio
async def test_resolve_no_args_errors():
    rid, err = await _resolve_single_source(["a"], source_id=None, sequence_number=None, season_number=None)
    assert rid is None and err


@pytest.mark.asyncio
async def test_execute_read_source_builds_narrative_and_registers_citation(monkeypatch):
    class Row(dict):
        def __getitem__(self, k):  # row["x"] and row.get("x")
            return super().__getitem__(k)

    src = {"id": "a", "title": "Ep 5", "summary": "the duel", "duration_seconds": 80 * 60, "language": "zh"}
    segs = [
        Row({"seq": 0, "start_s": 0.0, "end_s": 2.0, "speaker": "SPK_0", "text": "hello"}),
        Row({"seq": 1, "start_s": 112.0, "end_s": 114.0, "speaker": "SPK_1", "text": "world"}),
    ]

    async def fake_get_source(sid):  # noqa: ANN001
        return src

    async def fake_segments(sid):  # noqa: ANN001
        return segs

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    registry: dict[str, CitationRegistryEntry] = {}
    out = await execute_read_source(["a"], source_id="a", sequence_number=None, season_number=None, registry=registry)

    assert out["source_id"] == "a"
    assert "Ep 5" in out["_chunks"] and "the duel" in out["_chunks"]
    assert "hello" in out["_chunks"] and "@1:52" in out["_chunks"]  # 112s inline ts
    assert registry["a"].index == 1  # registered with next index


@pytest.mark.asyncio
async def test_read_source_reuses_find_passages_index(monkeypatch):
    """Shared registry, dedup by source_id: read_source on an already-registered
    source keeps the same [N] (spec §5.7)."""
    src = {"id": "a", "title": "Ep 5", "summary": "s", "duration_seconds": 60, "language": "zh"}

    async def fake_get_source(sid):  # noqa: ANN001
        return src

    async def fake_segments(sid):  # noqa: ANN001
        return []

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    registry = {"a": CitationRegistryEntry(index=2, source_id="a", title="Ep 5")}
    out = await execute_read_source(["a"], source_id="a", sequence_number=None, season_number=None, registry=registry)
    assert registry["a"].index == 2  # unchanged
    assert out["source_id"] == "a"


@pytest.mark.asyncio
async def test_read_source_empty_transcript_suppresses_header(monkeypatch):
    """No transcript segments → fact-only, header SUPPRESSED (spec §5.6/§16.3).
    The content header (title/summary/duration) must NOT leak — it is fabrication
    fuel with no body to frame."""
    src = {
        "id": "a",
        "title": "Secret Heist Episode",
        "summary": "the big robbery",
        "duration_seconds": 80 * 60,
        "language": "zh",
    }

    async def fake_get_source(sid):  # noqa: ANN001
        return src

    async def fake_segments(sid):  # noqa: ANN001
        return []

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    out = await execute_read_source(["a"], source_id="a", sequence_number=None, season_number=None, registry={})
    assert out["source_id"] == "a"
    assert "no transcript available" in out["_chunks"]
    # header content must NOT leak (no title / summary / duration / fence)
    assert "Secret Heist" not in out["_chunks"]
    assert "the big robbery" not in out["_chunks"]
    assert "=====" not in out["_chunks"]
