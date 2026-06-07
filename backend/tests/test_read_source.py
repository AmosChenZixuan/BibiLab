"""read_source tool: single-match resolution + narrative + citation binding."""

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


class Row(dict):
    def __getitem__(self, k):  # row["x"] and row.get("x")
        return super().__getitem__(k)


@pytest.mark.asyncio
async def test_resolve_by_citation_index_in_pool():
    rid, err = await _resolve_single_source(
        ["a", "b"],
        source_id="[1]",
        sequence_number=None,
        season_number=None,
        registry={"a": CitationRegistryEntry(index=1, source_id="a", title="Ep A")},
    )
    assert rid == "a" and err is None


@pytest.mark.asyncio
async def test_resolve_by_citation_index_bare_int_and_source_prefix():
    """Anchored regex accepts '[5]', '5', 'Source [5]', 'source 5'."""
    for variant in ("[5]", "5", "Source [5]", "source 5"):
        rid, err = await _resolve_single_source(
            ["a"],
            source_id=variant,
            sequence_number=None,
            season_number=None,
            registry={"a": CitationRegistryEntry(index=5, source_id="a", title="Ep 5")},
        )
        assert rid == "a" and err is None, (variant, rid, err)


@pytest.mark.asyncio
async def test_resolve_by_citation_index_bare_int_arg():
    """A non-string int arg (an LLM may emit source_id: 5 as a JSON number
    despite the string schema) is the cleanest index form — used directly,
    never round-tripped through the regex, never raised on."""
    rid, err = await _resolve_single_source(
        ["a"],
        source_id=5,
        sequence_number=None,
        season_number=None,
        registry={"a": CitationRegistryEntry(index=5, source_id="a", title="Ep 5")},
    )
    assert rid == "a" and err is None


@pytest.mark.asyncio
async def test_resolve_by_non_str_non_int_fails_loud():
    """A wholly wrong type (e.g. a list) must fail loud with the LLM-facing
    string, never raise — a raise aborts the SSE stream."""
    rid, err = await _resolve_single_source(
        ["a"],
        source_id=["5"],
        sequence_number=None,
        season_number=None,
        registry={"a": CitationRegistryEntry(index=5, source_id="a", title="Ep 5")},
    )
    assert rid is None
    assert "citation index" in err


@pytest.mark.asyncio
async def test_resolve_by_citation_index_unmapped_errors():
    """An [N] that exists syntactically but the registry has no entry for →
    'call find_passages first' error. Fail loud, no silent mis-resolution.
    The same path also covers the empty-registry case (read_source called
    before any find_passages)."""
    rid, err = await _resolve_single_source(
        ["a"],
        source_id="[9]",
        sequence_number=None,
        season_number=None,
        registry={"a": CitationRegistryEntry(index=1, source_id="a", title="Ep A")},
    )
    assert rid is None
    assert "[9]" in err and "find_passages" in err


@pytest.mark.asyncio
async def test_resolve_by_citation_index_deselected_fails_loud():
    """A [N] whose source_id is in the registry but NOT in the current pool
    (de-selected) must fail loud. Pool membership is the contract."""
    rid, err = await _resolve_single_source(
        ["b"],
        source_id="[1]",
        sequence_number=None,
        season_number=None,
        registry={
            "a": CitationRegistryEntry(index=1, source_id="a", title="Ep A"),
            "b": CitationRegistryEntry(index=2, source_id="b", title="Ep B"),
        },
    )
    assert rid is None
    assert "[1]" in err and "find_passages" in err


@pytest.mark.asyncio
async def test_resolve_by_non_index_string_errors():
    """A stray title like 'Episode 5 discussion' must not match the anchored
    regex — the resolver must fail loud, not silently parse the first int."""
    rid, err = await _resolve_single_source(
        ["a"],
        source_id="Episode 5 discussion",
        sequence_number=None,
        season_number=None,
        registry={"a": CitationRegistryEntry(index=1, source_id="a", title="Ep A")},
    )
    assert rid is None
    assert "citation index" in err


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

    # Pre-seed registry with [1] → "a" (as if find_passages had run earlier).
    registry = {"a": CitationRegistryEntry(index=1, source_id="a", title="Ep 5")}
    out = await execute_read_source(["a"], source_id="[1]", sequence_number=None, season_number=None, registry=registry)

    assert out["source_id"] == "a"
    assert out["tool_name"] == "read_source"
    assert out["source_title"] == "Ep 5"
    assert "Ep 5" in out["_chunks"] and "the duel" in out["_chunks"]
    assert "hello" in out["_chunks"] and "@1:52" in out["_chunks"]  # 112s inline ts
    assert registry["a"].index == 1  # reused, not re-allocated


@pytest.mark.asyncio
async def test_read_source_reuses_find_passages_index(monkeypatch):
    """Shared registry, dedup by source_id: read_source on an already-registered
    source keeps the same [N]."""
    src = {"id": "a", "title": "Ep 5", "summary": "s", "duration_seconds": 60, "language": "zh"}

    async def fake_get_source(sid):  # noqa: ANN001
        return src

    async def fake_segments(sid):  # noqa: ANN001
        return []

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    registry = {"a": CitationRegistryEntry(index=2, source_id="a", title="Ep 5")}
    out = await execute_read_source(["a"], source_id="[2]", sequence_number=None, season_number=None, registry=registry)
    assert registry["a"].index == 2  # unchanged
    assert out["source_id"] == "a"


@pytest.mark.asyncio
async def test_read_source_empty_transcript_suppresses_header(monkeypatch):
    """No transcript segments → fact-only, header SUPPRESSED.
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

    registry = {"a": CitationRegistryEntry(index=1, source_id="a", title="Secret Heist Episode")}
    out = await execute_read_source(["a"], source_id="[1]", sequence_number=None, season_number=None, registry=registry)
    assert out["source_id"] == "a"
    assert out["tool_name"] == "read_source"
    assert out["source_title"] == "Secret Heist Episode"
    assert "no transcript available" in out["_chunks"]
    # header content must NOT leak (no title / summary / duration / fence)
    assert "Secret Heist" not in out["_chunks"]
    assert "the big robbery" not in out["_chunks"]
    assert "=====" not in out["_chunks"]


@pytest.mark.asyncio
async def test_execute_read_source_resolves_by_facet_then_builds_narrative(monkeypatch):
    """Facet (sequence_number) must resolve to exactly one source, then the
    narrative is built for that source — same body as source_id path."""
    monkeypatch.setattr(
        chat_tools,
        "get_source_facets",
        _facets({"a": {"sequence_number": 4, "season_number": 1}, "b": {"sequence_number": 5, "season_number": 1}}),
    )

    src = {"id": "a", "title": "Ep 4", "summary": "the duel", "duration_seconds": 60 * 60, "language": "zh"}
    segs = [
        Row({"seq": 0, "start_s": 0.0, "end_s": 2.0, "speaker": "SPK_0", "text": "hello"}),
    ]

    async def fake_get_source(sid):
        return src

    async def fake_segments(sid):
        return segs

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    registry: dict[str, CitationRegistryEntry] = {}
    out = await execute_read_source(
        ["a", "b"], source_id=None, sequence_number=4, season_number=None, registry=registry
    )

    assert out["source_id"] == "a"  # resolved by facet to 'a'
    assert "Ep 4" in out["_chunks"]  # title in body
    assert "hello" in out["_chunks"]  # segment in body
    assert registry["a"].index == 1  # registered with next index


@pytest.mark.asyncio
async def test_execute_read_source_get_source_none_branch(monkeypatch):
    """Race or stale-resolution: get_source returns None after the resolve
    succeeded. Must return source_id=None + LLM-facing 'not found' — never
    raise (a raise aborts the SSE stream)."""
    monkeypatch.setattr(
        chat_tools,
        "get_source_facets",
        _facets({"a": {"sequence_number": 4, "season_number": 1}}),
    )

    async def fake_get_source(sid):
        return None  # source was deleted between resolve and read

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)

    out = await execute_read_source(["a"], source_id=None, sequence_number=4, season_number=None, registry={})

    assert out["source_id"] is None
    assert out["tool_name"] == "read_source"
    assert out["source_title"] == ""
    assert "not found" in out["_chunks"].lower()


@pytest.mark.asyncio
async def test_read_source_narrative_has_no_per_chunk_fence(monkeypatch):
    """read_source is a continuous transcript, NOT a fenced multi-source
    response. The non-empty body must NOT contain:
      - any `===== Source [` line (the find_passages per-source fence)
      - any `[N]:` per-chunk citation anchor (find_passages per-chunk anchor)
    Both belong to find_passages; read_source is a single-source read."""
    src = {"id": "a", "title": "Ep 4", "summary": "the duel", "duration_seconds": 60 * 60, "language": "zh"}
    segs = [
        Row({"seq": 0, "start_s": 0.0, "end_s": 2.0, "speaker": "SPK_0", "text": "first line"}),
        Row({"seq": 1, "start_s": 10.0, "end_s": 12.0, "speaker": "SPK_1", "text": "second line"}),
    ]

    async def fake_get_source(sid):
        return src

    async def fake_segments(sid):
        return segs

    monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
    monkeypatch.setattr(chat_tools, "get_transcript_segments", fake_segments)

    registry = {"a": CitationRegistryEntry(index=1, source_id="a", title="Ep 4")}
    out = await execute_read_source(["a"], source_id="[1]", sequence_number=None, season_number=None, registry=registry)
    body = out["_chunks"]

    # Continuous transcript fragments ARE present.
    assert "first line" in body and "second line" in body

    # Single ===== Source [N] fence (read_source is single-source). find_passages
    # would emit one ===== Source [N] header per hit source — multiple sources
    # → multiple fences. This pins the continuous-transcript contract.
    assert body.count("===== Source [") == 1, body
