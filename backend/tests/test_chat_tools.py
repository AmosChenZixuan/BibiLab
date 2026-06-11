"""Tests for the RAG v2 two-tool surface: find_passages + read_section + registry + provider expansion."""

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest


class TestFindPassagesToolSchema:
    def test_find_passages_tool_no_search_mode(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        props = FIND_PASSAGES_TOOL.parameters["properties"]
        assert "search_mode" not in props

    def test_find_passages_tool_no_source_filter(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        props = FIND_PASSAGES_TOOL.parameters["properties"]
        assert "source_filter" not in props

    def test_find_passages_tool_has_no_index_scope_params(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        props = FIND_PASSAGES_TOOL.parameters["properties"]
        assert "source_ids" not in props
        assert "exclude_source_ids" not in props

    def test_find_passages_tool_schema(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        props = FIND_PASSAGES_TOOL.parameters["properties"]
        assert "query" in props
        assert "expected_hits" not in props
        assert "sequence_number" in props
        assert "season_number" in props
        assert FIND_PASSAGES_TOOL.parameters["required"] == ["query"]

    def test_find_passages_tool_description_has_no_index_workflow(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        desc = FIND_PASSAGES_TOOL.description
        assert "exclude_source_ids" not in desc
        assert "source numbers" not in desc
        assert "Sources list" not in desc


class TestReadSectionToolSchema:
    def test_read_section_tool_name(self):
        from bibilab.pipeline.chat_tools import READ_SECTION_TOOL

        assert READ_SECTION_TOOL.name == "read_section"

    def test_read_section_tool_schema(self):
        from bibilab.pipeline.chat_tools import READ_SECTION_TOOL

        props = READ_SECTION_TOOL.parameters["properties"]
        assert "section_id" in props
        # Single required arg — section citation index from a find_passages result
        assert READ_SECTION_TOOL.parameters["required"] == ["section_id"]

    def test_read_section_tool_no_facet_args(self):
        from bibilab.pipeline.chat_tools import READ_SECTION_TOOL

        props = READ_SECTION_TOOL.parameters["properties"]
        # read_section resolves [N] only — no source_id/sequence/season args
        assert "source_id" not in props
        assert "sequence_number" not in props
        assert "season_number" not in props
        assert "query" not in props


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_read_section(self):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch.object(chat_tools, "execute_read_section", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"_chunks": "narrative", "section_id": "sec-1"}
            result = await execute_tool(
                tool_name="read_section",
                arguments={"section_id": "[5]"},
                source_ids=["a", "b"],
                cfg=cfg,
            )
            mock_exec.assert_awaited_once()
            assert result["section_id"] == "sec-1"

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_find_passages(self):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch.object(chat_tools, "execute_find_passages", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"_chunks": "x", "_turn_indices": [], "_raw_chunks": []}
            await execute_tool(
                tool_name="find_passages",
                arguments={"query": "q"},
                source_ids=["a"],
                cfg=cfg,
            )
            mock_exec.assert_awaited_once()
            # tool_name kwarg removed in v2
            call_kwargs = mock_exec.call_args.kwargs
            assert "tool_name" not in call_kwargs
            assert call_kwargs["query"] == "q"

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_raises(self):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool(
                tool_name="generate_report",  # retired
                arguments={},
                source_ids=[],
                cfg=cfg,
            )


class TestExecuteFindPassages:
    """Tests for execute_find_passages return shape and registry handling."""

    @pytest.mark.asyncio
    async def test_execute_find_passages_includes_query_in_result(self, monkeypatch):
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=1,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        result = await chat_tools.execute_find_passages(
            query="面食 种类",
            source_ids=["s1"],
            cfg=None,
        )

        assert result["query"] == "面食 种类"
        assert result["tool_name"] == "find_passages"
        assert "filter_miss" not in result

    @pytest.mark.asyncio
    async def test_execute_find_passages_drops_gate_neighbors_keys(self, monkeypatch):
        """v2 RetrievalResult has no dropped_by_gate/gate_margin/neighbors_pulled
        — execute_find_passages result must not include them."""
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=1,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
        result = await chat_tools.execute_find_passages(query="q", source_ids=["s1"], cfg=None)
        for stale in ("dropped_by_gate", "gate_margin", "neighbors_pulled", "mode"):
            assert stale not in result


@pytest.mark.asyncio
async def test_execute_find_passages_returns_raw_chunks_for_replay(monkeypatch):
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="verbatim text",
                    video_title="Video One",
                    timestamp_start=120.4,
                    timestamp_end=145.0,
                    source_id="s1",
                    distance=0.1,
                    score=0.9,
                    seg_start=0,
                    seg_end=0,
                ),
            ],
            candidates_evaluated=5,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id="s1", video_title="Video One", best_score=-0.9)],
        )

    section_rows = [
        {
            "id": "sec-1",
            "source_id": "s1",
            "seq": 0,
            "seg_start": 0,
            "seg_end": 0,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 200.0,
            "summary": "summary",
            "keywords": "[]",
        }
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=[]))

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}

    result = await chat_tools.execute_find_passages(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
    )

    assert "_raw_chunks" in result
    raw = result["_raw_chunks"]
    assert len(raw) == 1
    assert raw[0]["source_id"] == "s1"
    assert raw[0]["section_id"] == "sec-1"
    assert raw[0]["content"] == "verbatim text"
    assert raw[0]["video_title"] == "Video One"
    assert raw[0]["timestamp_start"] == 120.4
    assert raw[0]["timestamp_end"] == 145.0
    assert raw[0]["citation_index"] == 1
    assert raw[0]["chunk_id"] == "s1_120_145"


# AC2: CitationRegistryEntry gets timestamp_start, timestamp_end, rerank_score, preview
@pytest.mark.asyncio
async def test_citation_registry_entry_gets_chunk_fields_on_execute_find_passages(monkeypatch):
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="test content here",
                    video_title="Test Video",
                    timestamp_start=120.4,
                    timestamp_end=145.0,
                    source_id="s1",
                    distance=0.0,
                    score=0.95,
                    seg_start=0,
                    seg_end=0,
                ),
            ],
            candidates_evaluated=1,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id="s1", video_title="Test Video", best_score=-0.95)],
        )

    section_rows = [
        {
            "id": "sec-1",
            "source_id": "s1",
            "seq": 0,
            "seg_start": 0,
            "seg_end": 0,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 200.0,
            "summary": "summary",
            "keywords": "[]",
        }
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=[]))

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    registry: dict = {}

    await chat_tools.execute_find_passages(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
    )

    assert "sec-1" in registry
    entry = registry["sec-1"]
    assert entry.timestamp_start == 120.4
    assert entry.timestamp_end == 145.0
    assert entry.rerank_score == 0.95
    assert entry.preview == "test content here"


# AC2: multiple chunks from same source only populate first chunk's fields
@pytest.mark.asyncio
async def test_citation_registry_entry_uses_first_chunk_fields(monkeypatch):
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="first chunk",
                    video_title="V",
                    timestamp_start=10.0,
                    timestamp_end=20.0,
                    source_id="s1",
                    distance=0.0,
                    score=0.8,
                    seg_start=0,
                    seg_end=0,
                ),
                RetrievedChunk(
                    content="second chunk",
                    video_title="V",
                    timestamp_start=30.0,
                    timestamp_end=40.0,
                    source_id="s1",
                    distance=0.0,
                    score=0.9,
                    seg_start=1,
                    seg_end=1,
                ),
            ],
            candidates_evaluated=2,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id="s1", video_title="V", best_score=-0.9)],
        )

    section_rows = [
        {
            "id": "sec-1",
            "source_id": "s1",
            "seq": 0,
            "seg_start": 0,
            "seg_end": 1,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 50.0,
            "summary": "summary",
            "keywords": "[]",
        }
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=[]))

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    registry: dict = {}

    await chat_tools.execute_find_passages(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
    )

    entry = registry["sec-1"]
    # First chunk's timestamps, not second
    assert entry.timestamp_start == 10.0
    assert entry.timestamp_end == 40.0
    # First chunk's score
    assert entry.rerank_score == 0.8


def test_citation_entry_is_section_keyed_and_citable_defaults_false():
    from bibilab.pipeline.chat_tools import CitationRegistryEntry

    e = CitationRegistryEntry(index=1, section_id="sec-1", source_id="src-1", title="T")
    assert e.section_id == "sec-1"
    assert e.source_id == "src-1"
    assert e.seq is None
    assert e.citable is False  # outline-only until verbatim is shown
    assert e.chunk_ids == set()


class TestBuildToolBlockEntry:
    def test_build_tool_block_entry_find_passages_strips_internal_underscore_fields(self):
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        retrieve_result = {
            "query": "test",
            "tool_name": "find_passages",
            "candidates_evaluated": 5,
            "sources_with_hits": 2,
            "sources_total": 3,
            "source_coverage": [
                {"source_id": "s1", "title": "Video One"},
            ],
            "_chunks": "internal formatted string — must not be stored",
            "_turn_indices": [1],
        }
        raw_chunks = [
            {
                "source_id": "s1",
                "chunk_id": "v1_120_145",
                "content": "verbatim text",
                "video_title": "Video One",
                "timestamp_start": 120.4,
                "timestamp_end": 145.0,
                "citation_index": 1,
            },
        ]

        entry = build_tool_block_entry(
            tool_use_id="toolu_1",
            name="find_passages",
            arguments={"query": "test"},
            result=retrieve_result,
            raw_chunks=raw_chunks,
        )

        assert entry["tool_use_id"] == "toolu_1"
        assert entry["name"] == "find_passages"
        assert entry["arguments"] == {"query": "test"}
        assert "_chunks" not in entry["result"]
        assert "_turn_indices" not in entry["result"]
        assert entry["result"]["chunks"] == raw_chunks
        assert entry["result"]["summary"]["sources_total"] == 3

    def test_build_tool_block_entry_read_section_strips_internal_underscore_fields(self):
        """v2: read_section is never replayed/reseeded — strip its 30-150K
        token narrative from the persisted block. Only the summary remains."""
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        entry = build_tool_block_entry(
            tool_use_id="toolu_2",
            name="read_section",
            arguments={"section_id": "[5]"},
            result={
                "_chunks": "30-150K token narrative — must not be persisted",
                "section_id": "sec-1",
                "source_id": "a",
            },
            raw_chunks=None,
        )

        assert entry["name"] == "read_section"
        # The narrative MUST be stripped — never replayed, never reseeded.
        assert "_chunks" not in entry["result"]
        # Only the small scalars survive.
        assert entry["result"] == {"section_id": "sec-1", "source_id": "a"}


def test_expand_message_for_provider_text_only_passthrough_anthropic():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {"role": "assistant", "content": "Hi there"}
    out = expand_message_for_provider(msg, protocol="anthropic")
    assert out == [{"role": "assistant", "content": "Hi there"}]


def test_expand_message_for_provider_text_only_passthrough_openai():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {"role": "user", "content": "What's up"}
    out = expand_message_for_provider(msg, protocol="openai")
    assert out == [{"role": "user", "content": "What's up"}]


def test_expand_message_for_provider_empty_tool_blocks_passthrough():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {"role": "assistant", "content": "Hi", "tool_blocks": []}
    out = expand_message_for_provider(msg, protocol="anthropic")
    assert out == [{"role": "assistant", "content": "Hi"}]


def test_expand_message_for_provider_drops_retrieve_and_read_section_anthropic():
    """v2: find_passages + read_section tool exchanges are DROPPED
    from cross-turn replay (stale-context contamination). Only the assistant
    text survives."""
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {
        "role": "assistant",
        "content": "Answer [1]",
        "tool_blocks": [
            {
                "tool_use_id": "toolu_1",
                "name": "find_passages",
                "arguments": {"query": "q"},
                "result": {"chunks": [{"content": "x"}], "summary": {"sources_total": 1}},
            },
            {
                "tool_use_id": "toolu_2",
                "name": "read_section",
                "arguments": {"section_id": "[1]"},
                "result": {"_chunks": "big", "section_id": "sec-1", "source_id": "a"},
            },
        ],
    }
    out = expand_message_for_provider(msg, protocol="anthropic")

    # Both dropped → text-only fallback: single assistant message, no tool shape.
    assert len(out) == 1
    assert out[0] == {"role": "assistant", "content": "Answer [1]"}
    assert "tool_blocks" not in out[0]


def test_expand_message_for_provider_drops_find_passages_openai():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {
        "role": "assistant",
        "content": "Answer [1]",
        "tool_blocks": [
            {
                "tool_use_id": "call_1",
                "name": "find_passages",
                "arguments": {"query": "q"},
                "result": {"chunks": [{"content": "x"}], "summary": {"sources_total": 1}},
            }
        ],
    }
    out = expand_message_for_provider(msg, protocol="openai")

    assert len(out) == 1
    assert out[0] == {"role": "assistant", "content": "Answer [1]"}
    assert "tool_blocks" not in out[0]


class TestReseedCitationRegistry:
    def test_reseed_basic_single_chunk(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Answer [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "find_passages",
                        "arguments": {"query": "q"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 2,
                                    "citation_index": 1,
                                    "chunk_id": "v1_0_10",
                                    "video_title": "Video One",
                                    "content": "verbatim text",
                                }
                            ],
                            "summary": {"sources_total": 1},
                        },
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 1
        assert "sec-1" in registry
        entry = registry["sec-1"]
        assert entry.index == 1
        assert entry.section_id == "sec-1"
        assert entry.source_id == "s1"
        assert entry.title == "Video One"
        assert entry.seq == 2
        assert "v1_0_10" in entry.chunk_ids

    def test_reseed_accumulates_chunk_ids_for_same_section(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Answer",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "find_passages",
                        "arguments": {"query": "q"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 1,
                                    "citation_index": 1,
                                    "chunk_id": "v1_0_10",
                                    "video_title": "V1",
                                },
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 1,
                                    "citation_index": 1,
                                    "chunk_id": "v1_10_20",
                                    "video_title": "V1",
                                },
                            ],
                            "summary": {"sources_total": 1},
                        },
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 1
        entry = registry["sec-1"]
        assert "v1_0_10" in entry.chunk_ids
        assert "v1_10_20" in entry.chunk_ids
        assert len(entry.chunk_ids) == 2

    def test_reseed_separates_sections_within_same_source(self):
        """Two sections of the same source each get their own registry entry."""
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Answer",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "find_passages",
                        "arguments": {"query": "q"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 1,
                                    "citation_index": 1,
                                    "chunk_id": "v1_0_10",
                                    "video_title": "V1",
                                },
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-2",
                                    "section_seq": 2,
                                    "citation_index": 2,
                                    "chunk_id": "v1_50_60",
                                    "video_title": "V1",
                                },
                            ],
                            "summary": {"sources_total": 1},
                        },
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 2
        assert "sec-1" in registry
        assert "sec-2" in registry
        assert registry["sec-1"].index == 1
        assert registry["sec-2"].index == 2

    def test_reseed_skips_non_retrieve_blocks(self):
        """v2: read_section blocks do NOT feed the citation registry — they
        look up an already-registered section but never register a new one."""
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Read 5",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "read_section",
                        "arguments": {"section_id": "[5]"},
                        "result": {"_chunks": "...", "section_id": "sec-1", "source_id": "a"},
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 0

    def test_reseed_preserves_existing_entries(self):
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, reseed_citation_registry

        registry = {
            "sec-1": CitationRegistryEntry(
                index=1, section_id="sec-1", source_id="s1", title="V1", chunk_ids={"v1_0_10"}
            ),
        }
        history = [
            {
                "role": "assistant",
                "content": "More info [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "t2",
                        "name": "find_passages",
                        "arguments": {"query": "q2"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 1,
                                    "citation_index": 1,
                                    "chunk_id": "v1_50_60",
                                    "video_title": "V1",
                                }
                            ],
                            "summary": {"sources_total": 1},
                        },
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 1
        entry = registry["sec-1"]
        assert entry.index == 1
        assert "v1_0_10" in entry.chunk_ids
        assert "v1_50_60" in entry.chunk_ids
        assert len(entry.chunk_ids) == 2

    def test_reseed_handles_missing_citation_index(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Answer",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "find_passages",
                        "arguments": {"query": "q"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "section_id": "sec-1",
                                    "section_seq": 1,
                                    "chunk_id": "v1_0_10",
                                    "video_title": "V1",
                                },
                                {
                                    "source_id": "s2",
                                    "section_id": "sec-2",
                                    "section_seq": 1,
                                    "citation_index": 2,
                                    "chunk_id": "v2_0_10",
                                    "video_title": "V2",
                                },
                            ],
                            "summary": {"sources_total": 2},
                        },
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert "sec-1" not in registry
        assert "sec-2" in registry
        assert registry["sec-2"].index == 2

    def test_reseed_empty_history(self):
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, reseed_citation_registry

        registry: dict = {}
        reseed_citation_registry(registry, [])
        assert len(registry) == 0

        registry_with_entry = {
            "sec-1": CitationRegistryEntry(index=1, section_id="sec-1", source_id="s1", title="V1"),
        }
        reseed_citation_registry(registry_with_entry, [])
        assert len(registry_with_entry) == 1
        assert "sec-1" in registry_with_entry


class TestFacetInt:
    """_facet_int wraps the shared parse_facet_int, degrading unusable LLM
    args to None (never raises)."""

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (8, 8),
            ("8", 8),
            (8.0, 8),
            (" 8 ", 8),
            (1, 1),
            (None, None),
            (0, None),
            (-3, None),
            ("eight", None),
            ("", None),
            ("8.5", None),
            (8.5, None),
            (True, None),
            (False, None),
            ([8], None),
            ({"n": 8}, None),
        ],
    )
    def test_facet_int(self, val, expected):
        from bibilab.pipeline.chat_tools import _facet_int

        assert _facet_int(val, "sequence_number") == expected

    def test_unusable_value_logs_drop(self, caplog):
        from bibilab.pipeline.chat_tools import _facet_int

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            assert _facet_int(0, "sequence_number") is None
        assert "unusable, dropping predicate" in caplog.text


class TestFindPassagesFacetSchema:
    """find_passages schema exposes optional sequence_number/season_number."""

    def test_facet_params_present_and_typed(self):
        from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

        props = FIND_PASSAGES_TOOL.parameters["properties"]
        assert props["sequence_number"]["type"] == "integer"
        assert props["season_number"]["type"] == "integer"
        assert "series_name" not in props
        assert "sequence_number" not in FIND_PASSAGES_TOOL.parameters["required"]
        assert "season_number" not in FIND_PASSAGES_TOOL.parameters["required"]


class TestExecuteFindPassagesFacetScoping:
    """deterministic facet scoping with fail-open."""

    @staticmethod
    def _cfg():
        from bibilab.config import AIConfig, BibilabConfig

        return BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

    @staticmethod
    def _empty_result():
        from bibilab.pipeline.embed import RetrievalResult

        return RetrievalResult(
            chunks=[],
            source_coverage=[],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=3,
        )

    def _patch(self, monkeypatch, facets):
        """Patch retrieve (capture scoped_source_ids) and get_source_facets."""
        from bibilab.pipeline import chat_tools

        captured = {}

        async def fake_retrieve(query_text, source_ids, cfg, top_k, scoped_source_ids=None, **kw):
            captured["scoped"] = scoped_source_ids
            return self._empty_result()

        async def fake_get_source_facets(ids):
            return {k: v for k, v in facets.items() if k in ids}

        async def fake_get_segments_for_ranges(ranges):
            return []

        async def fake_get_sections(sid):
            return []

        async def fake_get_source(sid):
            return None

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_tools, "get_source_facets", fake_get_source_facets)
        monkeypatch.setattr(chat_tools, "get_segments_for_ranges", fake_get_segments_for_ranges)
        monkeypatch.setattr(chat_tools, "get_sections", fake_get_sections)
        monkeypatch.setattr(chat_tools, "get_source", fake_get_source)
        return captured

    @pytest.mark.asyncio
    async def test_sequence_number_scopes_pool(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 8, "season_number": None},
                "b": {"sequence_number": 9, "season_number": None},
                "c": {"sequence_number": 8, "season_number": None},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b", "c"],
            cfg=self._cfg(),
            sequence_number=8,
        )
        assert sorted(captured["scoped"]) == ["a", "c"]
        assert result["facet_scope"] == {
            "sequence_number": 8,
            "season_number": None,
            "matched_count": 2,
            "no_match": False,
        }
        assert result["scoped_pool_size"] == 3

    @pytest.mark.asyncio
    async def test_zero_match_fails_open_to_prefacet_pool(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 8, "season_number": None},
                "b": {"sequence_number": 9, "season_number": None},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=99,
        )
        assert captured["scoped"] is None
        assert result["facet_scope"]["no_match"] is True
        assert result["facet_scope"]["matched_count"] == 0

    @pytest.mark.asyncio
    async def test_no_facet_params_byte_identical(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(monkeypatch, {})
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
        )
        assert captured["scoped"] is None
        assert result["facet_scope"] == {
            "sequence_number": None,
            "season_number": None,
            "matched_count": None,
            "no_match": False,
        }

    @pytest.mark.asyncio
    async def test_no_match_prepends_note_to_llm_chunks(self, monkeypatch):
        """facet_scope.no_match true ⇒ _chunks carries the LLM-
        visible fact-only note (directive moved to the prompt)."""
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_MATCH_NOTE

        self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": None},
                "b": {"sequence_number": 2, "season_number": None},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=99,
        )
        assert result["facet_scope"]["no_match"] is True
        assert _NO_MATCH_NOTE in result["_chunks"]

    @pytest.mark.asyncio
    async def test_match_does_not_prepend_note(self, monkeypatch):
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_MATCH_NOTE

        self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": None},
                "b": {"sequence_number": 2, "season_number": None},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
        )
        assert result["facet_scope"]["no_match"] is False
        assert _NO_MATCH_NOTE not in result["_chunks"]

    @pytest.mark.asyncio
    async def test_season_omitted_does_not_filter(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": 2},
                "b": {"sequence_number": 1, "season_number": 5},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
        )
        assert sorted(captured["scoped"]) == ["a", "b"]
        assert result["facet_scope"]["matched_count"] == 2

    @pytest.mark.asyncio
    async def test_season_predicate_and_two_key_intersection(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": 2},
                "b": {"sequence_number": 1, "season_number": 5},
            },
        )
        result = await chat_tools.execute_find_passages(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
            season_number=2,
        )
        assert captured["scoped"] == ["a"]
        assert result["facet_scope"] == {
            "sequence_number": 1,
            "season_number": 2,
            "matched_count": 1,
            "no_match": False,
        }

    @pytest.mark.asyncio
    async def test_empty_result_emits_fact_only_no_coverage_fact(self, monkeypatch):
        """v2: zero-chunk find_passages emits the bare fact (no directive);
        the directive ('Tell the user … stop') moved to the prompt."""
        from bibilab.pipeline import chat_tools

        self._patch(monkeypatch, {})
        result = await chat_tools.execute_find_passages(
            query="什么是多元主义",
            source_ids=["s1", "s2", "s3"],
            cfg=self._cfg(),
        )
        assert result["_chunks"].startswith("find_passages found no relevant excerpts for this query.")


class TestExecuteToolFacetArgs:
    """execute_tool parses + coerces facet args from raw LLM arguments."""

    @staticmethod
    def _cfg():
        from bibilab.config import AIConfig, BibilabConfig

        return BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

    @pytest.mark.asyncio
    async def test_string_facet_arg_coerced_and_forwarded(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        seen = {}

        async def fake_execute_find_passages(**kwargs):
            seen.update(kwargs)
            return {"facet_scope": {}, "_chunks": "", "_turn_indices": [], "_raw_chunks": []}

        monkeypatch.setattr(chat_tools, "execute_find_passages", fake_execute_find_passages)

        await chat_tools.execute_tool(
            tool_name="find_passages",
            arguments={"query": "第八集", "sequence_number": "8"},
            source_ids=["a"],
            cfg=self._cfg(),
        )
        assert seen["sequence_number"] == 8
        assert seen["season_number"] is None

    @pytest.mark.asyncio
    async def test_non_numeric_facet_arg_dropped(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        seen = {}

        async def fake_execute_find_passages(**kwargs):
            seen.update(kwargs)
            return {"facet_scope": {}, "_chunks": "", "_turn_indices": [], "_raw_chunks": []}

        monkeypatch.setattr(chat_tools, "execute_find_passages", fake_execute_find_passages)

        await chat_tools.execute_tool(
            tool_name="find_passages",
            arguments={"query": "q", "sequence_number": "eight"},
            source_ids=["a"],
            cfg=self._cfg(),
        )
        assert seen["sequence_number"] is None


def test_section_for_seg_maps_by_containment():
    from bibilab.pipeline.chat_tools import _section_for_seg

    # sections: list of (section_id, seq, seg_start, seg_end)
    sections = [("a", 1, 0, 9), ("b", 2, 10, 19), ("c", 3, 20, 29)]
    assert _section_for_seg(sections, 0) == ("a", 1, 0, 9)
    assert _section_for_seg(sections, 9) == ("a", 1, 0, 9)
    assert _section_for_seg(sections, 10) == ("b", 2, 10, 19)
    assert _section_for_seg(sections, 25) == ("c", 3, 20, 29)
    assert _section_for_seg(sections, 99) is None  # out of range → None


# ---------------------------------------------------------------------------
# Task 3: section-keyed fence/header builders
# ---------------------------------------------------------------------------


def test_section_fence_header_shape():
    from bibilab.pipeline.chat_tools import CitationRegistryEntry, _build_section_fence_header

    e = CitationRegistryEntry(
        index=3,
        section_id="s3",
        source_id="src",
        title="My Show",
        seq=2,
        timestamp_start=600.0,
        timestamp_end=1200.0,
    )
    h = _build_section_fence_header(e)
    assert h == '===== [3] "My Show" · Section 2 (10:00–20:00) ====='


def test_build_fenced_sections_orders_by_index_and_includes_summary_then_chunks():
    from bibilab.pipeline.chat_tools import CitationRegistryEntry, _build_fenced_sections

    reg = {
        "s1": CitationRegistryEntry(
            index=1, section_id="s1", source_id="src", title="T", seq=1, timestamp_start=0.0, timestamp_end=60.0
        ),
        "s2": CitationRegistryEntry(
            index=2, section_id="s2", source_id="src", title="T", seq=2, timestamp_start=60.0, timestamp_end=120.0
        ),
    }
    summaries = {1: "Section one summary", 2: "Section two summary"}
    chunks_by_index = {1: [(0, 0, "[S1·SPK0 @0:01] hello")], 2: []}
    out = _build_fenced_sections(chunks_by_index, summaries, reg)
    # index order, header → summary → chunk body; section 2 has summary, no chunks
    assert out.index('[1] "T" · Section 1') < out.index('[2] "T" · Section 2')
    assert "Section one summary" in out
    assert "[S1·SPK0 @0:01] hello" in out
    assert "Section two summary" in out


def test_join_section_fragments_sorts_chronologically_and_marks_gaps():
    from bibilab.pipeline.chat_tools import _FRAGMENT_GAP, _join_section_fragments

    # Fragments arrive in rerank (out-of-spoken) order: seg 5 before seg 0.
    out = _join_section_fragments([(5, 6, "later"), (0, 1, "earlier")])
    # Chronological: earlier renders before later.
    assert out.index("earlier") < out.index("later")
    # Non-adjacent (seg 1 → seg 5) ⇒ gap marker between them.
    assert _FRAGMENT_GAP in out


def test_join_section_fragments_no_gap_when_adjacent():
    from bibilab.pipeline.chat_tools import _FRAGMENT_GAP, _join_section_fragments

    # seg_end 1 → next seg_start 2: contiguous, no transcript skipped.
    out = _join_section_fragments([(0, 1, "first"), (2, 3, "second")])
    assert _FRAGMENT_GAP not in out
    assert out == "first\nsecond"


def test_join_section_fragments_empty():
    from bibilab.pipeline.chat_tools import _join_section_fragments

    assert _join_section_fragments([]) == ""


@pytest.mark.asyncio
async def test_execute_find_passages_chunks_grouped_and_fenced(monkeypatch):
    """T4: _chunks groups by section (not source) + fences each section;
    _raw_chunks preserves the original interleaved rerank order (invariant).
    """
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import build_tool_block_entry
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="a1",
                    video_title="Video One",
                    timestamp_start=0.0,
                    timestamp_end=9.0,
                    source_id="v1",
                    distance=0.1,
                    score=0.9,
                    seg_start=0,
                    seg_end=0,
                ),
                RetrievedChunk(
                    content="b1",
                    video_title="Video Two",
                    timestamp_start=0.0,
                    timestamp_end=9.0,
                    source_id="v2",
                    distance=0.1,
                    score=0.8,
                    seg_start=0,
                    seg_end=0,
                ),
                RetrievedChunk(
                    content="a2",
                    video_title="Video One",
                    timestamp_start=9.0,
                    timestamp_end=18.0,
                    source_id="v1",
                    distance=0.2,
                    score=0.7,
                    seg_start=1,
                    seg_end=1,
                ),
                RetrievedChunk(
                    content="b2",
                    video_title="Video Two",
                    timestamp_start=9.0,
                    timestamp_end=18.0,
                    source_id="v2",
                    distance=0.2,
                    score=0.6,
                    seg_start=1,
                    seg_end=1,
                ),
            ],
            candidates_evaluated=4,
            sources_with_hits=2,
            sources_total=2,
            source_coverage=[
                SourceHit(source_id="v1", video_title="Video One", best_score=-0.9),
                SourceHit(source_id="v2", video_title="Video Two", best_score=-0.8),
            ],
        )

    section_rows = [
        {
            "id": f"sec-{sid}",
            "source_id": sid,
            "seq": 0,
            "seg_start": 0,
            "seg_end": 1,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 20.0,
            "summary": "summary",
            "keywords": "[]",
        }
        for sid in ("v1", "v2")
    ]

    async def fake_get_sections(sid):
        return [r for r in section_rows if r["source_id"] == sid]

    async def fake_get_segments_for_ranges(ranges):
        return []

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_sections", fake_get_sections)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", fake_get_segments_for_ranges)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}

    result = await chat_tools.execute_find_passages(
        query="q",
        source_ids=["v1", "v2"],
        cfg=cfg,
        registry=registry,
    )

    chunks = result["_chunks"]

    # Section fences present, both surfaces, ascending order.
    f1 = chunks.index('===== [1] "Video One" · Section 1')
    f2 = chunks.index('===== [2] "Video Two" · Section 1')
    assert f1 < f2

    # INVARIANT: _raw_chunks keeps original interleaved order + section-keyed indices.
    raw = result["_raw_chunks"]
    assert [r["citation_index"] for r in raw] == [1, 2, 1, 2]
    assert [r["content"] for r in raw] == ["a1", "b1", "a2", "b2"]
    # Each raw chunk carries its section_id / section_seq.
    for r in raw:
        assert r["section_id"] in ("sec-v1", "sec-v2")
        assert r["section_seq"] == 1

    # INVARIANT: fences must not leak into the persisted replay block.
    entry = build_tool_block_entry("tu1", "find_passages", {}, result, result["_raw_chunks"])
    assert "=====" not in json.dumps(entry, ensure_ascii=False)
    assert "_chunks" not in entry["result"]


@pytest.mark.asyncio
async def test_execute_find_passages_reconstructs_namespaced_turn_body(monkeypatch):
    """P4 Layer 5: displayed chunks with a seg-range get a speaker-turn body
    (grouped + time + S{N}·SPK{k}) reconstructed from transcript_segments, fed
    into both the fence body and the persisted preview."""
    from unittest.mock import AsyncMock

    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="raw-a1",
                    video_title="Video One",
                    timestamp_start=0.0,
                    timestamp_end=8.0,
                    source_id="v1",
                    distance=0.1,
                    score=0.9,
                    seg_start=0,
                    seg_end=1,
                ),
                RetrievedChunk(
                    content="raw-a2",
                    video_title="Video One",
                    timestamp_start=9.0,
                    timestamp_end=12.0,
                    source_id="v1",
                    distance=0.2,
                    score=0.7,
                    seg_start=2,
                    seg_end=2,
                ),
            ],
            candidates_evaluated=2,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id="v1", video_title="Video One", best_score=-0.9)],
        )

    seg_rows = [
        {"source_id": "v1", "seq": 0, "start_s": 0.0, "end_s": 4.0, "speaker": "SPK_0", "text": "你好。"},
        {"source_id": "v1", "seq": 1, "start_s": 5.0, "end_s": 8.0, "speaker": "SPK_1", "text": "再见。"},
        {"source_id": "v1", "seq": 2, "start_s": 9.0, "end_s": 12.0, "speaker": "SPK_0", "text": "结束。"},
    ]
    # Both chunks land in section 1 (seg range 0..29). T4 needs get_sections
    # to allocate the section; the section's seq is 0-based in DB, 1-based in
    # the registry/fence (see the offset in chat_tools.execute_find_passages).
    section_rows = [
        {
            "id": "sec-1",
            "source_id": "v1",
            "seq": 0,
            "seg_start": 0,
            "seg_end": 29,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 30.0,
            "summary": "section summary",
            "keywords": "[]",
        }
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=seg_rows))
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}

    result = await chat_tools.execute_find_passages(
        query="q",
        source_ids=["v1"],
        cfg=cfg,
        registry=registry,
    )

    chunks = result["_chunks"]
    assert "[S1·SPK0 @0:00] 你好。" in chunks
    assert "[S1·SPK1 @0:05] 再见。" in chunks
    assert "[S1·SPK0 @0:09] 结束。" in chunks
    assert "raw-a1" not in chunks
    assert "raw-a2" not in chunks

    # T4: registry is keyed by section_id, not source_id.
    assert "sec-1" in registry
    assert registry["sec-1"].source_id == "v1"
    assert registry["sec-1"].seq == 1
    assert registry["sec-1"].preview == "[S1·SPK0 @0:00] 你好。\n[S1·SPK1 @0:05] 再见。"


@pytest.mark.asyncio
async def test_execute_find_passages_orders_same_section_chronologically_with_gap(monkeypatch):
    """#464 end-to-end: two hits in one section returned in REVERSE rerank order
    (the later-seg fragment scores highest) render chronologically in the fence,
    with a gap marker between the non-adjacent fragments. Guards the wiring from
    chunk seg-range → _join_section_fragments — a transposed/wrong tuple arg at
    the append site would pass the helper's own unit tests but fail here."""
    from unittest.mock import AsyncMock

    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import _FRAGMENT_GAP
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                # rerank-top is the LATER fragment (seg 10-11) — out of spoken order.
                RetrievedChunk(
                    content="late",
                    video_title="V1",
                    timestamp_start=100.0,
                    timestamp_end=110.0,
                    source_id="v1",
                    distance=0.1,
                    score=0.9,
                    seg_start=10,
                    seg_end=11,
                ),
                RetrievedChunk(
                    content="early",
                    video_title="V1",
                    timestamp_start=0.0,
                    timestamp_end=8.0,
                    source_id="v1",
                    distance=0.2,
                    score=0.7,
                    seg_start=0,
                    seg_end=1,
                ),
            ],
            candidates_evaluated=2,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id="v1", video_title="V1", best_score=-0.9)],
        )

    # Both chunks nest in one section (seg 0..29).
    seg_rows = [
        {"source_id": "v1", "seq": 0, "start_s": 0.0, "end_s": 4.0, "speaker": "SPK_0", "text": "EARLY."},
        {"source_id": "v1", "seq": 1, "start_s": 5.0, "end_s": 8.0, "speaker": "SPK_0", "text": "early-two."},
        {"source_id": "v1", "seq": 10, "start_s": 100.0, "end_s": 105.0, "speaker": "SPK_0", "text": "LATE."},
        {"source_id": "v1", "seq": 11, "start_s": 106.0, "end_s": 110.0, "speaker": "SPK_0", "text": "late-two."},
    ]
    section_rows = [
        {
            "id": "sec-1",
            "source_id": "v1",
            "seq": 0,
            "seg_start": 0,
            "seg_end": 29,
            "token_count": 100,
            "timestamp_start": 0.0,
            "timestamp_end": 110.0,
            "summary": "summary",
            "keywords": "[]",
        }
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=seg_rows))
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""), backend=BackendConfig())

    result = await chat_tools.execute_find_passages(query="q", source_ids=["v1"], cfg=cfg, registry={})
    chunks = result["_chunks"]

    # Chronological despite LATE being the rerank-top chunk, with the gap marker
    # between the two non-adjacent (seg 1 → seg 10) fragments.
    assert chunks.index("EARLY.") < chunks.index(_FRAGMENT_GAP) < chunks.index("LATE.")


# ---------------------------------------------------------------------------
# Task 4: execute_find_passages groups hits by section
# ---------------------------------------------------------------------------


async def _build_two_section_source(list_id: str, **source_overrides) -> str:
    """Seed a 2-section, 30-segment source with topic-mentioning sentences."""
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests.factories import SourceFactory

    segs = [
        WhisperSegment(
            start=float(i),
            end=float(i + 1),
            text=f"sentence {i} about the topic discussed",
            speaker="SPK_0",
        )
        for i in range(30)
    ]
    sections = [
        Section(seg_start=0, seg_end=14, token_count=100, timestamp_start=0.0, timestamp_end=15.0),
        Section(seg_start=15, seg_end=29, token_count=100, timestamp_start=15.0, timestamp_end=30.0),
    ]
    digests = [
        SectionDigest(summary="sec1 summary about the topic", keywords=["k1"]),
        SectionDigest(summary="sec2 summary about the topic", keywords=["k2"]),
    ]
    return await SourceFactory.build(
        list_id,
        segments=segs,
        sections=sections,
        section_digests=digests,
        **source_overrides,
    )


@pytest.mark.asyncio
async def test_find_passages_registers_sections_not_sources(tmp_bibilab_home, monkeypatch):
    """T4: registry keys are section_id, each entry carries seq + citable=True
    when the section's verbatim is shown, and the rendered body uses the
    section-keyed fence. Real sections come from the factory; retrieve +
    get_segments are mocked so the test stays self-contained."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.db import bootstrap_db, create_list, get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    # Real DB with a 2-section source. 30 segments split 0-14 + 15-29.
    await bootstrap_db()
    await create_list("L1", "Test List", "2025-01-01T00:00:00Z")
    source_id = await _build_two_section_source(
        "L1",
        video_id="BVtwoSection",
        title="Two Section Video",
    )
    section_rows = await get_sections(source_id)
    section_ids = [r["id"] for r in section_rows]

    # retrieve returns one chunk in each section.
    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="raw-s1",
                    video_title="Two Section Video",
                    timestamp_start=0.0,
                    timestamp_end=14.0,
                    source_id=source_id,
                    distance=0.1,
                    score=0.9,
                    seg_start=0,
                    seg_end=14,
                ),
                RetrievedChunk(
                    content="raw-s2",
                    video_title="Two Section Video",
                    timestamp_start=15.0,
                    timestamp_end=29.0,
                    source_id=source_id,
                    distance=0.2,
                    score=0.8,
                    seg_start=15,
                    seg_end=29,
                ),
            ],
            candidates_evaluated=2,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id=source_id, video_title="Two Section Video", best_score=-0.9)],
        )

    seg_rows = [
        {
            "source_id": source_id,
            "seq": i,
            "start_s": float(i),
            "end_s": float(i + 1),
            "speaker": "SPK_0",
            "text": f"sentence {i} about the topic discussed",
        }
        for i in range(30)
    ]

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=seg_rows))
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}
    result = await chat_tools.execute_find_passages(
        query="the topic discussed",
        source_ids=[source_id],
        cfg=cfg,
        registry=registry,
    )

    # registry keyed by section_id (not source_id), each entry carries seq.
    assert set(registry.keys()) <= set(section_ids)
    assert source_id not in registry, "registry must be section-keyed, not source-keyed"
    for entry in registry.values():
        assert entry.section_id in section_ids
        assert entry.seq in (1, 2)
    # at least one surfaced section shows verbatim → citable
    assert any(e.citable for e in registry.values())
    # _chunks contains a section fence
    assert "· Section" in result["_chunks"]
    # _raw_chunks carries section_id / section_seq for downstream consumers
    for r in result["_raw_chunks"]:
        assert r["section_id"] in section_ids
        assert r["section_seq"] in (1, 2)
    # section_coverage replaces source_coverage in the public result
    assert "section_coverage" in result
    assert "source_coverage" not in result


# ---------------------------------------------------------------------------
# Task 5: facet → full section OUTLINE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_passages_facet_emits_full_outline(tmp_bibilab_home, monkeypatch):
    """T5: a facet match expands to the matched source's FULL section outline
    — every section gets its own [N], even when the query hit only one
    section's chunks. Outline-only sections stay citable=False until drilled."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.db import bootstrap_db, create_list, get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, SourceHit

    # Real DB with a 2-section source. 30 segments split 0-14 + 15-29.
    await bootstrap_db()
    await create_list("L1", "Test List", "2025-01-01T00:00:00Z")
    source_id = await _build_two_section_source(
        "L1",
        video_id="BVfacetSource",
        title="Facet Match Video",
        sequence_number=1,
    )
    section_rows = await get_sections(source_id)
    section_ids = [r["id"] for r in section_rows]

    # retrieve returns only ONE chunk, in section 1. The outline expansion
    # must still register section 2 (the other section of the matched source)
    # with citable=False.
    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                # chunk in section 1 only (seg range 0..14)
            ],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=1,
            source_coverage=[SourceHit(source_id=source_id, video_title="Facet Match Video", best_score=-0.9)],
        )

    # get_source_facets must report sequence_number=1 on the seeded source.
    async def fake_get_source_facets(ids):
        return {sid: {"sequence_number": 1, "season_number": None} for sid in ids}

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_source_facets", fake_get_source_facets)
    # The T4 path also calls get_sections / get_segments_for_ranges; stub them
    # so the test stays self-contained (real sections already loaded above).
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=[]))

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}
    result = await chat_tools.execute_find_passages(
        query="anything",
        source_ids=[source_id],
        cfg=cfg,
        registry=registry,
        sequence_number=1,
    )

    # Every section of the matched source is registered (outline expansion).
    assert {e.section_id for e in registry.values()} == set(section_ids)
    # facet_scope reports the matched count.
    assert result["facet_scope"]["matched_count"] == 1
    # Outline-only sections (no verbatim) stay citable=False.
    outline_only = [e for e in registry.values() if not e.citable]
    assert outline_only, "at least one section must be outline-only (no chunk hit)"
    # All section summaries appear in the fenced body.
    assert "sec1 summary about the topic" in result["_chunks"]
    assert "sec2 summary about the topic" in result["_chunks"]
    # section_coverage includes the outline entries (both sections, not just the hit one).
    cov_ids = {row["section_id"] for row in result["section_coverage"]}
    assert cov_ids == set(section_ids)
    # Outline fence headers render the REAL section span, not 0:00–0:00.
    # Regression for the outline-timestamp defect (entry.timestamp_start was
    # left None for outline-only allocations, so the fence showed 0:00–0:00
    # and the LLM/ledger were misled about the section's location).
    assert "0:00–0:15" in result["_chunks"], "section 1 fence should show its real range, not 0:00–0:00"
    assert "0:15–0:30" in result["_chunks"], "section 2 fence should show its real range, not 0:00–0:00"
    assert "0:00–0:00" not in result["_chunks"], "no outline fence should render 0:00–0:00"


# ---------------------------------------------------------------------------
# read_section: [N]=section, bounded verbatim
# ---------------------------------------------------------------------------


class TestReadSectionUnitPaths:
    """Pure unit tests for execute_read_section (no DB)."""

    @pytest.mark.asyncio
    async def test_read_section_unknown_index_fails_loud_not_raises(self):
        """[9] with no matching registry entry → LLM-facing error, no raise.
        A raise would abort the SSE stream — read_section must return."""
        from bibilab.pipeline.chat_tools import execute_read_section

        out = await execute_read_section(source_ids=["src"], section_id="[9]", registry={})
        assert out["section_id"] is None
        assert out["source_id"] is None
        assert out["source_title"] == ""
        assert out["tool_name"] == "read_section"
        assert "No section [9]" in out["_chunks"]

    @pytest.mark.asyncio
    async def test_read_section_non_index_string_fails_loud(self):
        """A stray title like 'Episode 5 discussion' must fail loud — the
        anchored regex must NOT silently parse the first int."""
        from bibilab.pipeline.chat_tools import execute_read_section

        out = await execute_read_section(source_ids=["a"], section_id="Episode 5 discussion", registry={})
        assert out["section_id"] is None
        assert "section_id must be a citation index" in out["_chunks"]

    @pytest.mark.asyncio
    async def test_read_section_pool_membership_enforced(self):
        """[N] whose section's source_id is NOT in the current source_ids
        pool (de-selected) must fail loud."""
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, execute_read_section

        reg = {"sec-a": CitationRegistryEntry(index=1, section_id="sec-a", source_id="a", title="Ep A", seq=1)}
        # Pool is ["b"] — source "a" is not in the pool
        out = await execute_read_section(source_ids=["b"], section_id="[1]", registry=reg)
        assert out["section_id"] is None
        assert "No section [1]" in out["_chunks"]


@pytest.mark.asyncio
async def test_read_section_returns_bounded_verbatim(tmp_bibilab_home, monkeypatch):
    """Integration: read_section resolves a [N] citation index from the
    registry to a single section, returns that section's bounded verbatim
    transcript (format_turns with citation_index), and flips citable=True."""
    from bibilab.db import bootstrap_db, create_list, get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests.factories import SourceFactory

    # 2-section source; section 2 has 5 segs (15-19), section 1 has 15 (0-14)
    await bootstrap_db()
    await create_list("L1", "Test List", "2025-01-01T00:00:00Z")

    segs = [
        WhisperSegment(
            start=float(i),
            end=float(i + 1),
            text=f"sentence {i} about the topic",
            speaker="SPK_0",
        )
        for i in range(20)
    ]
    sections = [
        Section(seg_start=0, seg_end=14, token_count=100, timestamp_start=0.0, timestamp_end=15.0),
        Section(seg_start=15, seg_end=19, token_count=50, timestamp_start=15.0, timestamp_end=20.0),
    ]
    digests = [
        SectionDigest(summary="sec1", keywords=["k1"]),
        SectionDigest(summary="sec2", keywords=["k2"]),
    ]
    source_id = await SourceFactory.build(
        "L1",
        video_id="BVreadSection",
        title="Read Section Video",
        segments=segs,
        sections=sections,
        section_digests=digests,
    )
    section_rows = await get_sections(source_id)
    section_ids = [r["id"] for r in section_rows]

    # Pre-seed registry with [5] → section 2 (the second section, seq=2 1-based).
    reg = {
        section_ids[1]: chat_tools.CitationRegistryEntry(
            index=5,
            section_id=section_ids[1],
            source_id=source_id,
            title="Read Section Video",
            seq=2,
        )
    }

    out = await chat_tools.execute_read_section(source_ids=[source_id], section_id="[5]", registry=reg)

    assert out["tool_name"] == "read_section"
    assert out["section_id"] == section_ids[1]
    assert out["source_id"] == source_id
    assert out["source_title"] == "Read Section Video"
    # Verbatim body uses citation_index=5 (S5 prefix), bounded to section 2's segs.
    body = out["_chunks"]
    assert "[S5·SPK0" in body
    # Section 2's segs (15..19) are present, section 1's are NOT.
    assert "sentence 19" in body and "sentence 15" in body
    assert "sentence 0" not in body and "sentence 14" not in body
    # Drilling flips the section's citable flag.
    assert reg[section_ids[1]].citable is True
    # Header is the section fence format (matches T3's _build_section_fence_header).
    assert '===== [5] "Read Section Video" · Section 2' in body


@pytest.mark.asyncio
async def test_read_section_section_not_found_branch(monkeypatch):
    """Section row vanished between find_passages and read_section (e.g. an
    out-of-band ingest rewrite). Return the LLM-facing fact, never raise."""
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import execute_read_section

    reg = {
        "sec-ghost": chat_tools.CitationRegistryEntry(
            index=2, section_id="sec-ghost", source_id="src", title="Ghost", seq=1
        )
    }

    async def fake_get_sections(sid):  # noqa: ANN001
        return []  # no sections left

    monkeypatch.setattr(chat_tools, "get_sections", fake_get_sections)
    out = await execute_read_section(source_ids=["src"], section_id="[2]", registry=reg)
    assert "not found" in out["_chunks"].lower() or "section [2]" in out["_chunks"]
    # Spec returns the entry IDs unconditionally (citable flip on success path
    # does not depend on narrative content), so structural ids are populated.
    assert out["section_id"] == "sec-ghost"
    assert out["source_id"] == "src"
    # Spec: citable is flipped unconditionally on the success path.
    assert reg["sec-ghost"].citable is True


@pytest.mark.asyncio
async def test_read_section_empty_segments_branch(monkeypatch):
    """Section exists but the segments were never written (e.g. transcript
    failed mid-pipeline). Return fact-only, suppress the header."""
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import execute_read_section

    reg = {
        "sec-empty": chat_tools.CitationRegistryEntry(
            index=3, section_id="sec-empty", source_id="src", title="Empty", seq=1
        )
    }

    async def fake_get_sections(sid):  # noqa: ANN001
        return [
            {
                "id": "sec-empty",
                "source_id": "src",
                "seq": 0,
                "seg_start": 0,
                "seg_end": 5,
                "token_count": 10,
                "timestamp_start": 0.0,
                "timestamp_end": 6.0,
                "summary": "x",
                "keywords": "[]",
            }
        ]

    async def fake_get_segments_for_ranges(ranges):  # noqa: ANN001
        return []

    monkeypatch.setattr(chat_tools, "get_sections", fake_get_sections)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", fake_get_segments_for_ranges)
    out = await execute_read_section(source_ids=["src"], section_id="[3]", registry=reg)
    assert "no transcript available" in out["_chunks"]
    # Title must NOT leak — the header is suppressed when there's no body.
    assert "Empty" not in out["_chunks"]
    assert "=====" not in out["_chunks"]
    # Spec: citable is flipped unconditionally on the success path.
    assert reg["sec-empty"].citable is True


@pytest.mark.asyncio
async def test_execute_tool_dispatches_read_section_args(monkeypatch):
    """execute_tool routes read_section through execute_read_section, passing
    section_id (no facet coercion)."""
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import execute_tool

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

    seen = {}

    async def fake_execute_read_section(**kwargs):
        seen.update(kwargs)
        return {"_chunks": "narr", "section_id": "sec-1"}

    monkeypatch.setattr(chat_tools, "execute_read_section", fake_execute_read_section)
    await execute_tool(
        tool_name="read_section",
        arguments={"section_id": "[3]"},
        source_ids=["a"],
        cfg=cfg,
    )
    assert seen["section_id"] == "[3]"
    assert seen["registry"] is None or isinstance(seen["registry"], dict)
    # No facet coercion — read_section takes [N] only.
    assert "sequence_number" not in seen
    assert "season_number" not in seen
