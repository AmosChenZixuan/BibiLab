"""Tests for incremental citation parser."""

import json

from bibilab.pipeline.chat_tools import CitationRegistryEntry
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta


def _e(index: int, source_id: str = "", section_id: str | None = None, citable: bool = True) -> CitationRegistryEntry:
    return CitationRegistryEntry(
        index=index,
        section_id=section_id or f"sec-{index}",
        source_id=source_id or f"s{index}",
        citable=citable,
    )


def _lk(*entries: CitationRegistryEntry) -> dict[int, CitationRegistryEntry]:
    return {e.index: e for e in entries}


class TestCompleteTokens:
    def test_single_citation(self):
        events, buf = parse_delta("see [1] there", "", _lk(_e(1)))
        assert len(events) == 3
        assert events[0].type == "delta" and events[0].content == "see "
        assert events[1].type == "citation"
        assert json.loads(events[1].content)["index"] == 1
        assert events[2].type == "delta" and events[2].content == " there"
        assert buf == ""

    def test_multiple_in_one_delta(self):
        events, _ = parse_delta("[1] and [2]", "", _lk(_e(1), _e(2)))
        types = [e.type for e in events]
        assert types == ["citation", "delta", "citation"]

    def test_no_citations_passthrough(self):
        events, buf = parse_delta("plain text", "", {})
        assert len(events) == 1
        assert events[0].type == "delta" and events[0].content == "plain text"

    def test_repeated_index_two_citation_events(self):
        events, _ = parse_delta("[1]...[1]", "", _lk(_e(1)))
        assert len([e for e in events if e.type == "citation"]) == 2

    def test_out_of_range_emitted_as_text(self):
        events, _ = parse_delta("see [7]", "", _lk(_e(1), _e(2)))
        assert all(e.type == "delta" for e in events)
        combined = "".join(e.content or "" for e in events)
        assert "[7]" in combined


class TestBoundaries:
    def test_bracket_then_number(self):
        _, buf1 = parse_delta("see [", "", _lk(_e(1)))
        assert buf1 == "["
        events2, buf2 = parse_delta("1] there", buf1, _lk(_e(1)))
        assert any(e.type == "citation" for e in events2)
        assert buf2 == ""

    def test_number_then_bracket(self):
        events1, buf1 = parse_delta("see [1", "", _lk(_e(1)))
        assert buf1 == "[1"
        events2, buf2 = parse_delta("] there", buf1, _lk(_e(1)))
        assert any(e.type == "citation" for e in events2)

    def test_number_split_mid_digit(self):
        """[1 + 23] → buffer has [123] → matches index 123 if in registry"""
        _, buf1 = parse_delta("x [1", "", _lk(_e(1), _e(123)))
        assert buf1 == "[1"
        events2, _ = parse_delta("23] y", buf1, _lk(_e(1), _e(123)))
        assert any(e.type == "citation" for e in events2)

    def test_two_complete_adjacent(self):
        events, buf = parse_delta("[1][2]", "", _lk(_e(1), _e(2)))
        assert len([e for e in events if e.type == "citation"]) == 2
        assert buf == ""


class TestMalformed:
    def test_text_in_brackets_passthrough(self):
        events, _ = parse_delta("see [abc]", "", _lk(_e(1)))
        combined = "".join(e.content or "" for e in events)
        assert "[abc]" in combined

    def test_spaced_brackets_passthrough(self):
        events, _ = parse_delta("see [ 1 ]", "", _lk(_e(1)))
        assert all(e.type == "delta" for e in events)

    def test_empty_brackets_passthrough(self):
        events, _ = parse_delta("see []", "", _lk(_e(1)))
        combined = "".join(e.content or "" for e in events)
        assert "[]" in combined


class TestFlushBuffer:
    def test_flush_unclosed(self):
        events = flush_buffer("trailing [1")
        assert len(events) == 1
        assert events[0].type == "delta"
        assert events[0].content == "trailing [1"

    def test_flush_empty(self):
        assert flush_buffer("") == []


def _citations(events: list) -> list[int]:
    return [json.loads(e.content)["index"] for e in events if e.type == "citation"]


def _delta_text(events: list) -> str:
    return "".join(e.content or "" for e in events if e.type == "delta")


class TestNonCanonicalWrappers:
    """AC1 — D1 wrappers normalize to the same citation event as [N]."""

    def test_bracket_source_label(self):
        events, buf = parse_delta("see [Source 1] there", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "see  there"
        assert buf == ""

    def test_paren_source_label(self):
        events, _ = parse_delta("see (Source 1) there", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "see  there"

    def test_fullwidth_paren_source_label(self):
        events, _ = parse_delta("see （Source 1） there", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "see  there"

    def test_bracket_chinese_label(self):
        events, _ = parse_delta("见[来源1]处", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "见处"

    def test_fullwidth_paren_chinese_label(self):
        events, _ = parse_delta("见（来源1）处", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "见处"

    def test_halfwidth_paren_chinese_label(self):
        events, _ = parse_delta("见(来源1)处", "", _lk(_e(1)))
        assert _citations(events) == [1]

    def test_lenticular_bare(self):
        events, _ = parse_delta("见【1】处", "", _lk(_e(1)))
        assert _citations(events) == [1]
        assert _delta_text(events) == "见处"

    def test_lenticular_chinese_label(self):
        events, _ = parse_delta("见【来源1】处", "", _lk(_e(1)))
        assert _citations(events) == [1]

    def test_source_label_case_insensitive(self):
        events, _ = parse_delta("see [source 1] and [SOURCE 1]", "", _lk(_e(1)))
        assert _citations(events) == [1, 1]

    def test_source_label_no_space(self):
        events, _ = parse_delta("see [Source1] there", "", _lk(_e(1)))
        assert _citations(events) == [1]

    def test_canonical_still_works(self):
        events, _ = parse_delta("see [1] there", "", _lk(_e(1)))
        assert _citations(events) == [1]


class TestMultiIndex:
    """AC2 — D2 multi-index: one citation event per index, in order, no text between."""

    def test_comma_ascii(self):
        events, _ = parse_delta("see [1,2]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1, 2]
        assert _delta_text(events) == "see "

    def test_comma_space(self):
        events, _ = parse_delta("see [1, 2]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1, 2]
        assert _delta_text(events) == "see "

    def test_ideographic_comma_fullwidth(self):
        events, _ = parse_delta("见【1，2】", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1, 2]
        assert _delta_text(events) == "见"

    def test_enumeration_comma(self):
        events, _ = parse_delta("见[1、2]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1, 2]

    def test_multi_index_no_delta_between(self):
        events, _ = parse_delta("[1,2]", "", _lk(_e(1), _e(2)))
        types = [e.type for e in events]
        assert types == ["citation", "citation"]


class TestMultiIndexOutOfRange:
    """AC3/D3 — per-index handling: known→citation, unknown→own numeric text delta."""

    def test_single_out_of_range_unchanged(self):
        events, _ = parse_delta("see [Source 7]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == []
        assert "[Source 7]" in _delta_text(events)

    def test_multi_index_partial_known(self):
        events, _ = parse_delta("[1,7]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1]
        # unknown index emitted as its own numeric text delta
        assert "7" in _delta_text(events)


class TestNonCanonicalFalsePositives:
    """AC3/D4 — no digit run in D1/D2 shape → no match, text passes through."""

    def test_source_word_in_prose(self):
        events, _ = parse_delta("Source code is open", "", _lk(_e(1)))
        assert _citations(events) == []
        assert _delta_text(events) == "Source code is open"

    def test_bare_chinese_label_no_digit(self):
        events, _ = parse_delta("参考来源很多", "", _lk(_e(1)))
        assert _citations(events) == []
        assert _delta_text(events) == "参考来源很多"

    def test_spaced_canonical_bracket_still_passthrough(self):
        events, _ = parse_delta("see [ 1 ]", "", _lk(_e(1)))
        assert _citations(events) == []
        assert "[ 1 ]" in _delta_text(events)


class TestNonCanonicalPartialBuffer:
    """AC4/D5 — non-canonical wrapper split across deltas resolves to one event."""

    def test_fullwidth_paren_split(self):
        events1, buf1 = parse_delta("see （来源", "", _lk(_e(1)))
        assert all(e.type == "delta" for e in events1)
        events2, buf2 = parse_delta("1）done", buf1, _lk(_e(1)))
        assert _citations(events2) == [1]
        assert buf2 == ""

    def test_lenticular_multi_index_split(self):
        events1, buf1 = parse_delta("x 【1", "", _lk(_e(1), _e(2)))
        events2, buf2 = parse_delta("，2】y", buf1, _lk(_e(1), _e(2)))
        assert _citations(events1) + _citations(events2) == [1, 2]
        assert buf2 == ""

    def test_bracket_source_label_split(self):
        events1, buf1 = parse_delta("see [Source ", "", _lk(_e(1)))
        events2, _ = parse_delta("1] done", buf1, _lk(_e(1)))
        assert _citations(events1) + _citations(events2) == [1]

    def test_flush_unclosed_non_canonical_is_text(self):
        _, buf1 = parse_delta("tail （来源", "", _lk(_e(1)))
        events = flush_buffer(buf1)
        assert all(e.type == "delta" for e in events)
        assert "来源" in "".join(e.content or "" for e in events)

    def test_non_viable_prefix_flushed_as_text(self):
        # "(abc" is not a viable prefix of any D1/D2 token → flush, don't hold
        events, buf = parse_delta("see (abc", "", _lk(_e(1)))
        assert _delta_text(events) == "see (abc"
        assert buf == ""


class TestTimestampSuffix:
    """Optional @M:SS clock-timestamp suffix after the index is silently
    consumed — the LLM copies it from the fence header / turn lines, which
    render time as @M:SS (or @H:MM:SS), ranges joined with an en-dash."""

    def test_bare_with_timestamp(self):
        events, buf = parse_delta("see [2 @0:20] there", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "see  there"
        assert buf == ""

    def test_endash_range(self):
        # The exact shape reported: index + @M:SS–M:SS range with an en-dash.
        events, _ = parse_delta("see [2 @0:43–5:10] there", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "see  there"

    def test_hyphen_range(self):
        events, _ = parse_delta("[2 @0:43-5:10]", "", _lk(_e(2)))
        assert _citations(events) == [2]

    def test_hms_timestamp(self):
        events, _ = parse_delta("[2 @1:05:30]", "", _lk(_e(2)))
        assert _citations(events) == [2]

    def test_no_space_before_at(self):
        events, _ = parse_delta("[2@0:20]", "", _lk(_e(2)))
        assert _citations(events) == [2]

    def test_source_label_with_timestamp(self):
        events, _ = parse_delta("see [Source 2 @0:20] there", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "see  there"

    def test_chinese_label_with_timestamp(self):
        events, _ = parse_delta("见[来源2 @0:20]处", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "见处"

    def test_multi_index_with_timestamp(self):
        events, _ = parse_delta("see [1, 2 @0:20]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == [1, 2]
        assert _delta_text(events) == "see "

    def test_lenticular_with_timestamp(self):
        events, _ = parse_delta("见【2 @0:20】处", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "见处"

    def test_no_timestamp_still_works(self):
        events, _ = parse_delta("see [2] there", "", _lk(_e(2)))
        assert _citations(events) == [2]
        assert _delta_text(events) == "see  there"

    def test_out_of_range_with_timestamp_stays_text(self):
        events, _ = parse_delta("see [7 @0:20]", "", _lk(_e(1), _e(2)))
        assert _citations(events) == []
        assert "[7 @0:20]" in _delta_text(events)

    def test_timestamp_split_across_deltas(self):
        # A clock suffix straddling a delta boundary must be held, not leaked.
        events1, buf1 = parse_delta("see [2 @0", "", _lk(_e(2)))
        assert _citations(events1) == []
        events2, buf2 = parse_delta(":20] there", buf1, _lk(_e(2)))
        assert _citations(events2) == [2]
        assert _delta_text(events1) + _delta_text(events2) == "see  there"
        assert buf2 == ""


class TestSectionPayload:
    """T7 — citation event payload gains section_id + timestamp_start
    (additive). Old fields (index, source_id, chunk_ids) preserved."""

    def test_citation_event_carries_section_and_timestamp(self):
        reg = {
            1: CitationRegistryEntry(
                index=1,
                section_id="sec-1",
                source_id="src-1",
                title="T",
                seq=2,
                citable=True,
                timestamp_start=42.0,
            )
        }
        events, _ = parse_delta("see [1] here", "", reg)
        cite = next(e for e in events if e.type == "citation")
        payload = json.loads(cite.content)
        assert payload == {
            "index": 1,
            "section_id": "sec-1",
            "source_id": "src-1",
            "timestamp_start": 42.0,
            "chunk_ids": [],
        }

    def test_citation_event_preserves_legacy_fields(self):
        """All three pre-existing payload keys must remain present."""
        reg = {
            1: CitationRegistryEntry(
                index=1,
                section_id="sec-1",
                source_id="src-1",
                title="T",
                citable=True,
            )
        }
        events, _ = parse_delta("[1]", "", reg)
        cite = next(e for e in events if e.type == "citation")
        payload = json.loads(cite.content)
        assert "index" in payload
        assert "source_id" in payload
        assert "chunk_ids" in payload

    def test_outline_only_section_emits_text_not_citation(self):
        """citable=False (outline-only) registry entries must NOT emit a citation
        event — the [N] marker is stripped to text instead. The LLM often
        references an outline section it never drilled into."""
        reg = {
            1: CitationRegistryEntry(
                index=1,
                section_id="s",
                source_id="src",
                title="T",
                seq=1,
                citable=False,
            )
        }
        events, _ = parse_delta("ref [1]", "", reg)
        assert all(e.type != "citation" for e in events)
        combined = "".join(e.content or "" for e in events if e.type == "delta")
        assert "[1]" in combined
