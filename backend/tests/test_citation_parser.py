"""Tests for incremental citation parser."""

import json

from bibilab.pipeline.chat_tools import CitationRegistryEntry
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta


def _e(index: int, source_id: str = "") -> CitationRegistryEntry:
    return CitationRegistryEntry(index=index, source_id=source_id or f"s{index}")


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
