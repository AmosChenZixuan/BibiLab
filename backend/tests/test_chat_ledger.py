"""RAG ledger reconstruction.

The ledger a chat turn persists is not what retrieval returned: coverage is
narrowed to the sections the assistant actually cited, and each surviving
section's context entry is rebuilt from the citation registry. These tests pin
that transformation directly, without driving an SSE turn.
"""

from __future__ import annotations

import pytest

from bibilab.pipeline.chat_ledger import build_rag_ledger, sum_truncation
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
    # the emitted set is turn-wide, not per-call. Ordering is not asserted here;
    # the per-call ordering logic runs inside this loop body, so the dedicated
    # order tests already cover it on a list long enough to discriminate.
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


class _SectionId(str):
    """A section id whose hash is pinned.

    Set iteration follows hash order, and CPython randomizes string hashing per
    process. With ordinary ids an unordered implementation reproduces the
    expected order under some seeds, so the regression stops being caught
    without the test ever going red. Pinning the hash makes the unordered path
    emit one known-wrong order on every run, whatever PYTHONHASHSEED is.

    Equality stays str's, so `==` and assertion output read as plain strings.
    The hash deliberately does not, which breaks the hash/eq contract: a
    `_SectionId` and an equal plain str land in different dict and set buckets.
    Within one fixture always build ids through `_sid`, never mix in a bare
    literal — a lookup by the wrong kind fails silently rather than raising.
    """

    _hash: int

    def __new__(cls, value: str, hash_: int) -> "_SectionId":
        obj = super().__new__(cls, value)
        obj._hash = hash_
        return obj

    def __hash__(self) -> int:
        return self._hash


# Hashes ascend in lexicographic order, which the rerank order below
# deliberately is not: an unordered implementation can only emit
# s1, s2, s3, … so producing the rerank order cannot happen by chance.
_SECTION_HASH = {"s1": 0, "s2": 1, "s3": 2, "s4": 3, "s5": 4, "s7": 5, "s8": 6, "s9": 7}


def _sid(value: str) -> _SectionId:
    return _SectionId(value, _SECTION_HASH[value])


# A rerank order that is neither alphabetical nor sorted.
_RERANK_ORDER = tuple(_sid(v) for v in ("s5", "s2", "s9", "s1", "s7", "s3", "s8", "s4"))


def _rerank_registry() -> dict[str, CitationRegistryEntry]:
    return {sid: _entry(sid, i + 1) for i, sid in enumerate(_RERANK_ORDER)}


def test_context_preserves_rerank_order_when_nothing_is_narrowed():
    # section_coverage arrives most-relevant-first; context[] is persisted and
    # rendered in array order, so it has to carry that same order out.
    calls = build_rag_ledger(
        retrieve_calls=[_retrieve_call(*_RERANK_ORDER)],
        read_section_calls=[],
        content_blocks=[],
        citation_registry=_rerank_registry(),
    )

    assert [c["section_id"] for c in calls[0]["context"]] == list(_RERANK_ORDER)


def test_context_preserves_rerank_order_through_narrowing():
    registry = _rerank_registry()
    cited = tuple(_sid(v) for v in ("s9", "s1", "s3", "s8", "s5", "s7"))

    calls = build_rag_ledger(
        retrieve_calls=[_retrieve_call(*_RERANK_ORDER)],
        read_section_calls=[],
        content_blocks=[_citation_block(registry[sid].index) for sid in cited],
        citation_registry=registry,
    )

    assert [c["section_id"] for c in calls[0]["context"]] == ["s5", "s9", "s1", "s7", "s3", "s8"]


def test_read_section_title_backfilled_via_section_id():
    rs = {"tool_name": "read_section", "section_id": "7", "source_id": "src-other", "source_title": ""}

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={"7": _entry("7", 1, title="Section Seven")},
    )

    assert calls[0]["source_title"] == "Section Seven"


@pytest.mark.parametrize("section_id", ["", "sec-absent"])
def test_read_section_title_ignores_source_id_keyed_entry(section_id):
    """The lookup is section-keyed only — a registry entry that happens to sit
    under a source id is not a match, whether the row's section id is missing
    or merely absent from the registry."""
    rs = {"tool_name": "read_section", "section_id": section_id, "source_id": "legacy-src", "source_title": ""}
    legacy = _entry("legacy-src", 1, source_id="legacy-src", title="Legacy Source")

    calls = build_rag_ledger(
        retrieve_calls=[],
        read_section_calls=[rs],
        content_blocks=[],
        citation_registry={"legacy-src": legacy},
    )

    assert calls[0]["source_title"] == ""


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


def test_sum_truncation_sums_pairs_and_tokens_but_maxes_worst_drop():
    """Two find_passages calls in one turn: truncated_pairs/tokens_dropped sum
    across calls, but worst_drop takes the max — not last-write-wins, and not a
    literal sum (which would no longer mean "largest single-pair loss")."""
    calls = [
        {"truncated_pairs": 2, "tokens_dropped": 30, "worst_drop": 20},
        {"truncated_pairs": 1, "tokens_dropped": 15, "worst_drop": 15},
    ]

    result = sum_truncation(calls)

    assert result == {"truncated_pairs": 3, "tokens_dropped": 45, "worst_drop": 20}


def test_sum_truncation_worst_drop_max_survives_a_clean_call_in_the_mix():
    """A call that truncated nothing (worst_drop=0) must not pull the turn-level
    max down — max() over [20, 0] must still pick 20."""
    calls = [
        {"truncated_pairs": 2, "tokens_dropped": 30, "worst_drop": 20},
        {"truncated_pairs": 0, "tokens_dropped": 0, "worst_drop": 0},
    ]

    result = sum_truncation(calls)

    assert result == {"truncated_pairs": 2, "tokens_dropped": 30, "worst_drop": 20}


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [{"truncated_pairs": 0, "tokens_dropped": 0, "worst_drop": 0}],
        [{}],
    ],
    ids=["no-calls", "zeroed-call", "missing-keys"],
)
def test_sum_truncation_reports_nothing_when_no_truncation(calls):
    assert sum_truncation(calls) == {}
