"""RAG ledger reconstruction.

The ledger a chat turn persists is not what retrieval returned: coverage is
narrowed to the sections the assistant actually cited, and each surviving
section's context entry is rebuilt from the citation registry. These tests pin
that transformation directly, without driving an SSE turn.
"""

from __future__ import annotations

import pytest

from bibilab.pipeline.chat_ledger import build_rag_ledger
from bibilab.pipeline.chat_tools import CitationRegistryEntry


def _entry(section_id: str, index: int, **overrides) -> CitationRegistryEntry:
    kwargs = {
        "index": index,
        "section_id": section_id,
        "source_id": f"src-{section_id}",
        "title": f"Title {section_id}",
        "seq": index,
        "first_chunk_id": f"chunk-{section_id}",
        "timestamp_start": 10.0 * index,
        "timestamp_end": 10.0 * index + 5.0,
        "rerank_score": 0.5 + index,
        "preview": f"preview {section_id}",
    }
    kwargs.update(overrides)
    return CitationRegistryEntry(**kwargs)


def _retrieve_call(*section_ids: str) -> dict:
    return {
        "query": "q",
        "tool_name": "find_passages",
        "section_coverage": [{"section_id": sid} for sid in section_ids],
    }


def _citation_block(index: int) -> dict:
    return {"type": "citation", "index": index}


@pytest.mark.parametrize(
    "content_blocks",
    [[], [{"type": "text", "text": "no citations here"}]],
    ids=["no-blocks", "text-only-blocks"],
)
def test_no_citations_leaves_coverage_intact_and_still_rebuilds_context(content_blocks):
    call = _retrieve_call("1", "2", "3")
    registry = {"1": _entry("1", 1), "2": _entry("2", 2), "3": _entry("3", 3)}

    calls = build_rag_ledger(
        retrieve_calls=[call],
        read_section_calls=[],
        content_blocks=content_blocks,
        citation_registry=registry,
    )

    assert [s["section_id"] for s in calls[0]["section_coverage"]] == ["1", "2", "3"]
    assert {c["section_id"] for c in calls[0]["context"]} == {"1", "2", "3"}


def test_partial_citations_narrow_coverage_to_the_cited_subset():
    call = _retrieve_call("1", "2", "3")
    registry = {"1": _entry("1", 1), "2": _entry("2", 2), "3": _entry("3", 3)}

    calls = build_rag_ledger(
        retrieve_calls=[call],
        read_section_calls=[],
        content_blocks=[_citation_block(1), _citation_block(3)],
        citation_registry=registry,
    )

    assert {s["section_id"] for s in calls[0]["section_coverage"]} == {"1", "3"}
    assert {c["section_id"] for c in calls[0]["context"]} == {"1", "3"}


def test_all_cited_keeps_coverage_and_rebuilds_every_registry_field():
    call = _retrieve_call("1", "2")
    registry = {"1": _entry("1", 1), "2": _entry("2", 2)}

    calls = build_rag_ledger(
        retrieve_calls=[call],
        read_section_calls=[],
        content_blocks=[_citation_block(1), _citation_block(2)],
        citation_registry=registry,
    )

    assert {s["section_id"] for s in calls[0]["section_coverage"]} == {"1", "2"}
    by_id = {c["section_id"]: c for c in calls[0]["context"]}
    assert by_id["2"] == {
        "section_id": "2",
        "section_seq": 2,
        "chunk_id": "chunk-2",
        "citation_index": 2,
        "source_id": "src-2",
        "source_title": "Title 2",
        "timestamp_start": 20.0,
        "timestamp_end": 25.0,
        "rerank_score": 2.5,
        "preview": "preview 2",
    }


def test_parallel_retrieve_calls_share_one_emitted_set():
    # Subject decomposition issues several find_passages calls in one turn. A
    # citation earned by one call still narrows every other call's coverage —
    # the emitted set is turn-wide, not per-call.
    call_a = _retrieve_call("1", "2")
    call_b = _retrieve_call("3", "4")
    registry = {str(i): _entry(str(i), i) for i in (1, 2, 3, 4)}

    calls = build_rag_ledger(
        retrieve_calls=[call_a, call_b],
        read_section_calls=[],
        content_blocks=[_citation_block(2)],
        citation_registry=registry,
    )

    assert [s["section_id"] for s in calls[0]["section_coverage"]] == ["2"]
    assert calls[1]["section_coverage"] == []
    assert [c["section_id"] for c in calls[0]["context"]] == ["2"]
    assert calls[1]["context"] == []


def test_covered_section_absent_from_registry_is_skipped_in_context():
    call = _retrieve_call("1", "ghost")

    calls = build_rag_ledger(
        retrieve_calls=[call],
        read_section_calls=[],
        content_blocks=[],
        citation_registry={"1": _entry("1", 1)},
    )

    assert {s["section_id"] for s in calls[0]["section_coverage"]} == {"1", "ghost"}
    assert [c["section_id"] for c in calls[0]["context"]] == ["1"]


def test_read_section_title_backfilled_via_section_id():
    rs = {"tool_name": "read_section", "section_id": "7", "source_id": "src-other", "source_title": ""}

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={"7": _entry("7", 1, title="Section Seven")},
    )

    assert calls[0]["source_title"] == "Section Seven"


def test_read_section_title_falls_back_to_source_id_lookup():
    rs = {"tool_name": "read_section", "section_id": "", "source_id": "legacy-src", "source_title": ""}
    legacy = _entry("legacy-src", 1, source_id="legacy-src", title="Legacy Source")

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={"legacy-src": legacy},
    )

    assert calls[0]["source_title"] == "Legacy Source"


def test_read_section_title_unresolvable_is_left_alone():
    rs = {"tool_name": "read_section", "section_id": "9", "source_id": "src-9", "source_title": ""}

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={},
    )

    assert calls[0]["source_title"] == ""


def test_read_section_existing_title_is_not_overwritten():
    rs = {"tool_name": "read_section", "section_id": "7", "source_id": "src-7", "source_title": "From tool"}

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={"7": _entry("7", 1, title="From registry")},
    )

    assert calls[0]["source_title"] == "From tool"


def test_read_section_context_is_an_empty_list_not_a_synthetic_entry():
    # The frontend ledger branches on the empty array to render a "read in full"
    # affordance; a zeroed entry would render as "0:00 / 0.00".
    rs = {"tool_name": "read_section", "section_id": "7", "source_id": "src-7", "source_title": "T"}

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[_citation_block(1)],
        citation_registry={"7": _entry("7", 1)},
    )

    assert calls[0]["context"] == []


def test_retrieve_calls_precede_read_section_calls():
    calls = build_rag_ledger(
        retrieve_calls=[_retrieve_call("1")],
        read_section_calls=[
            {"tool_name": "read_section", "section_id": "1", "source_id": "src-1", "source_title": "T"}
        ],
        content_blocks=[],
        citation_registry={"1": _entry("1", 1)},
    )

    assert [c["tool_name"] for c in calls] == ["find_passages", "read_section"]
