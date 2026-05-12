"""Tests for generate_report tool execution."""

from unittest.mock import AsyncMock, patch

import pytest


class TestRetrieveToolSchema:
    def test_retrieve_tool_no_search_mode(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "search_mode" not in props

    def test_retrieve_tool_has_source_filter(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "source_filter" in props
        sf = props["source_filter"]
        assert sf["type"] == "object"
        assert "title_contains" in sf["properties"]
        assert sf["properties"]["title_contains"]["type"] == "string"

    def test_retrieve_tool_has_expected_hits(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "expected_hits" in props
        assert props["expected_hits"]["enum"] == ["one", "few", "many"]

    def test_retrieve_tool_required_is_query_only(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        required = RETRIEVE_TOOL.parameters["required"]
        assert required == ["query"]

    def test_retrieve_tool_description_guides_source_filter(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        assert "source_filter" in desc
        assert "title_contains" in desc or "episode" in desc.lower()


class TestGenerateReportToolDefinition:
    def test_generate_report_tool_schema(self):
        from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL

        assert GENERATE_REPORT_TOOL.name == "generate_report"
        assert "type" in GENERATE_REPORT_TOOL.parameters["properties"]
        assert "prompt" in GENERATE_REPORT_TOOL.parameters["properties"]
        required = GENERATE_REPORT_TOOL.parameters.get("required", [])
        assert "type" in required
        assert "prompt" in required

    def test_generate_report_tool_type_enum(self):
        from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL

        type_prop = GENERATE_REPORT_TOOL.parameters["properties"]["type"]
        assert type_prop.get("enum") == ["brief", "study_guide", "blog_post", "custom_report"]


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_execute_generate_report_creates_artifact_job(self, tmp_bibilab_home):
        from bibilab.pipeline.chat_tools import execute_generate_report

        with patch("bibilab.pipeline.chat_tools.create_job", new_callable=AsyncMock) as mock_create_job:
            mock_create_job.return_value = "job-uuid-123"

            result = await execute_generate_report(
                list_id="list-uuid-1",
                artifact_type="study_guide",
                prompt="summarize the main points",
                source_ids=["src-1", "src-2"],
                ui_lang="en",
            )

            mock_create_job.assert_called_once()
            call_kwargs = mock_create_job.call_args.kwargs
            assert call_kwargs["type"] == "artifact"
            meta = call_kwargs["meta"]
            assert meta["list_id"] == "list-uuid-1"
            assert meta["type"] == "study_guide"
            assert meta["prompt"] == "summarize the main points"
            assert meta["source_ids"] == ["src-1", "src-2"]
            assert meta["ui_lang"] == "en"
            assert "artifact_id" in meta

            assert result["artifact_id"] == meta["artifact_id"]
            assert result["name"] == "study_guide"
            assert result["type"] == "study_guide"

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_generate_report(self, tmp_bibilab_home):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.execute_generate_report", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"artifact_id": "a1", "name": "brief", "type": "brief"}

            result = await execute_tool(
                tool_name="generate_report",
                arguments={"type": "brief", "prompt": "quick summary"},
                list_id="list-1",
                source_ids=["s1"],
                ui_lang="en",
                cfg=cfg,
            )

            mock_exec.assert_called_once_with(
                list_id="list-1",
                artifact_type="brief",
                prompt="quick summary",
                source_ids=["s1"],
                ui_lang="en",
            )
            assert result["artifact_id"] == "a1"

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_query_list_metadata(self, tmp_bibilab_home):
        from unittest.mock import AsyncMock, patch

        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch(
            "bibilab.pipeline.chat_tools.execute_query_list_metadata",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = {"count": 8}

            result = await execute_tool(
                tool_name="query_list_metadata",
                arguments={"query_type": "count"},
                list_id="list-1",
                source_ids=["s1", "s2"],
                ui_lang="en",
                cfg=cfg,
            )

            mock_exec.assert_awaited_once_with(source_ids=["s1", "s2"], query_type="count")
            assert result == {"count": 8}

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_raises(self, tmp_bibilab_home):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline.chat_tools import execute_tool

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool(
                tool_name="unknown_tool",
                arguments={},
                list_id="list-1",
                source_ids=[],
                ui_lang="en",
                cfg=cfg,
            )


class TestQueryListMetadataToolDefinition:
    def test_tool_schema(self):
        from bibilab.pipeline.chat_tools import QUERY_LIST_METADATA_TOOL

        assert QUERY_LIST_METADATA_TOOL.name == "query_list_metadata"
        props = QUERY_LIST_METADATA_TOOL.parameters["properties"]
        assert "query_type" in props
        assert props["query_type"]["enum"] == ["count", "longest", "languages"]
        assert QUERY_LIST_METADATA_TOOL.parameters["required"] == ["query_type"]


class TestExecuteQueryListMetadata:
    @pytest.mark.asyncio
    async def test_count_returns_count_dict(self, tmp_bibilab_home):
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.count_sources", new_callable=AsyncMock) as m:
            m.return_value = 8
            result = await execute_query_list_metadata(["s1", "s2"], "count")
        assert result == {"count": 8}
        m.assert_awaited_once_with(["s1", "s2"])

    @pytest.mark.asyncio
    async def test_longest_returns_title_and_duration(self, tmp_bibilab_home):
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.longest_source", new_callable=AsyncMock) as m:
            m.return_value = {"title": "Foo", "duration_seconds": 3600}
            result = await execute_query_list_metadata(["s1"], "longest")
        assert result == {"title": "Foo", "duration_seconds": 3600}

    @pytest.mark.asyncio
    async def test_longest_empty_returns_title_none_duration_none(self, tmp_bibilab_home):
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.longest_source", new_callable=AsyncMock) as m:
            m.return_value = None
            result = await execute_query_list_metadata([], "longest")
        assert result == {"title": None, "duration_seconds": None}

    @pytest.mark.asyncio
    async def test_languages_returns_breakdown_dict(self, tmp_bibilab_home):
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.language_breakdown", new_callable=AsyncMock) as m:
            m.return_value = {"zh": 5, "en": 3, "unknown": 1}
            result = await execute_query_list_metadata(["s1", "s2"], "languages")
        assert result == {"languages": {"zh": 5, "en": 3, "unknown": 1}}

    @pytest.mark.asyncio
    async def test_unknown_query_type_falls_back_to_count(self, tmp_bibilab_home, caplog):
        import logging
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.count_sources", new_callable=AsyncMock) as m:
            m.return_value = 4
            with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
                result = await execute_query_list_metadata(["s1"], "garbage")

        assert result == {"count": 4}
        assert "Unknown query_type" in caplog.text


class TestFormatChunkForLLM:
    def test_format_with_index(self):
        from bibilab.pipeline.chat_tools import _format_chunk_for_llm

        result = _format_chunk_for_llm(
            {"title": "V", "start": 120.0, "end": 145.0, "content": "hi"},
            index=3,
        )
        assert result == '[3 @ 120s-145s]: "hi"'

    def test_truncates_floats(self):
        from bibilab.pipeline.chat_tools import _format_chunk_for_llm

        result = _format_chunk_for_llm(
            {"title": "V", "start": 10.9, "end": 20.1, "content": "x"},
            index=1,
        )
        assert result == '[1 @ 10s-20s]: "x"'


class TestExecuteRetrieve:
    """Tests for execute_retrieve return shape and registry handling."""

    @pytest.mark.asyncio
    async def test_execute_retrieve_includes_query_in_result(self, monkeypatch):
        """The result returned by execute_retrieve must include the original query."""
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

        result = await chat_tools.execute_retrieve(
            query="面食 种类",
            search_mode="breadth",
            source_ids=["s1"],
            cfg=None,
        )

        assert result["query"] == "面食 种类"
        assert result["search_mode"] == "breadth"


@pytest.mark.asyncio
async def test_execute_retrieve_returns_raw_chunks_for_replay(monkeypatch):
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, params):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="verbatim text",
                    video_title="Video One",
                    timestamp_start=120.4,
                    timestamp_end=145.0,
                    video_id="v1",
                    distance=0.1,
                    score=0.9,
                ),
            ],
            candidates_evaluated=5,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(video_id="v1", video_title="Video One", best_score=-0.9)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}
    source_map = {"v1": "s1"}

    result = await chat_tools.execute_retrieve(
        query="test",
        search_mode="factual",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
        source_map=source_map,
    )

    assert "_raw_chunks" in result
    raw = result["_raw_chunks"]
    assert len(raw) == 1
    assert raw[0]["source_id"] == "s1"
    assert raw[0]["content"] == "verbatim text"
    assert raw[0]["video_title"] == "Video One"
    assert raw[0]["timestamp_start"] == 120.4
    assert raw[0]["timestamp_end"] == 145.0
    assert raw[0]["citation_index"] == 1
    assert raw[0]["chunk_id"] == "v1_120_145"


class TestBuildSourceHeaders:
    def test_single(self):
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, _build_source_headers

        r = {"s1": CitationRegistryEntry(index=1, source_id="s1", title="My Video")}
        assert _build_source_headers(r) == 'Source [1]: "My Video"'

    def test_sorted_by_index(self):
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, _build_source_headers

        r = {
            "sb": CitationRegistryEntry(index=2, source_id="sb", title="B"),
            "sa": CitationRegistryEntry(index=1, source_id="sa", title="A"),
        }
        result = _build_source_headers(r)
        lines = result.split("\n")
        assert lines[0] == 'Source [1]: "A"'
        assert lines[1] == 'Source [2]: "B"'


class TestBuildGroundingPrompt:
    def test_build_grounding_prompt_includes_response_language_prefix(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="zh")
        assert prompt.startswith("Respond in zh.")

    def test_build_grounding_prompt_localizes_fallback_sentence(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="zh")
        # The fallback rule must reference the response_language, not hard-coded English.
        assert "say so in zh" in prompt
        # And the old hard-coded English string must be gone.
        assert "The provided sources do not cover this topic" not in prompt

    def test_build_grounding_prompt_has_fresh_retrieve_directive(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="en")
        assert "Each new user question requires a fresh `retrieve` call" in prompt
        assert "Do not infer answers from your own prior text" in prompt
        assert "When in doubt, retrieve." in prompt

    def test_build_grounding_prompt_has_verbatim_proper_noun_rule(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="en")
        assert "exact spelling from the retrieved excerpts" in prompt
        assert "Do not paraphrase or translate proper nouns" in prompt

    def test_build_grounding_prompt_has_no_real_world_parallels_rule(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="en")
        assert "start your answer with what the excerpts say" in prompt
        assert "Describe the concept as the source presents it" in prompt


class TestRetrieveToolDescription:
    def test_retrieve_tool_description_drops_rephrasing_exclusion(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        # The "rephrasing" exclusion conflated user rephrasing with model rephrasing
        # and caused short content questions to skip retrieval.
        assert "rephrasing" not in desc

    def test_retrieve_tool_description_biases_toward_retrieve_for_short_questions(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        assert "short or vague" in desc or "even short" in desc
        assert "content question" in desc


class TestBuildToolBlockEntry:
    def test_build_tool_block_entry_retrieve_strips_internal_underscore_fields(self):
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        retrieve_result = {
            "query": "test",
            "search_mode": "factual",
            "candidates_evaluated": 5,
            "sources_with_hits": 2,
            "sources_total": 3,
            "source_coverage": [
                {"source_id": "s1", "video_id": "v1", "title": "Video One"},
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
            name="retrieve",
            arguments={"query": "test", "search_mode": "factual"},
            result=retrieve_result,
            raw_chunks=raw_chunks,
        )

        assert entry["tool_use_id"] == "toolu_1"
        assert entry["name"] == "retrieve"
        assert entry["arguments"] == {"query": "test", "search_mode": "factual"}
        assert "_chunks" not in entry["result"]
        assert "_turn_indices" not in entry["result"]
        assert entry["result"]["chunks"] == raw_chunks
        assert entry["result"]["summary"]["sources_total"] == 3

    def test_build_tool_block_entry_non_retrieve_preserves_result_as_is(self):
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        entry = build_tool_block_entry(
            tool_use_id="toolu_2",
            name="query_list_metadata",
            arguments={"query_type": "count"},
            result={"count": 5},
            raw_chunks=None,
        )

        assert entry["tool_use_id"] == "toolu_2"
        assert entry["name"] == "query_list_metadata"
        assert entry["result"] == {"count": 5}


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


def test_expand_message_for_provider_anthropic_shape():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {
        "role": "assistant",
        "content": "Answer [1]",
        "tool_blocks": [
            {
                "tool_use_id": "toolu_1",
                "name": "retrieve",
                "arguments": {"query": "q", "search_mode": "factual"},
                "result": {"chunks": [{"content": "x"}], "summary": {"sources_total": 1}},
            }
        ],
    }
    out = expand_message_for_provider(msg, protocol="anthropic")

    assert len(out) == 2
    assert out[0]["role"] == "assistant"
    assert out[0]["content"][0] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "retrieve",
        "input": {"query": "q", "search_mode": "factual"},
    }
    assert out[0]["content"][-1] == {"type": "text", "text": "Answer [1]"}
    assert out[1]["role"] == "user"
    assert out[1]["content"][0]["type"] == "tool_result"
    assert out[1]["content"][0]["tool_use_id"] == "toolu_1"


def test_expand_message_for_provider_openai_shape():
    from bibilab.pipeline.chat_tools import expand_message_for_provider

    msg = {
        "role": "assistant",
        "content": "Answer [1]",
        "tool_blocks": [
            {
                "tool_use_id": "call_1",
                "name": "retrieve",
                "arguments": {"query": "q", "search_mode": "factual"},
                "result": {"chunks": [{"content": "x"}], "summary": {"sources_total": 1}},
            }
        ],
    }
    out = expand_message_for_provider(msg, protocol="openai")

    # OpenAI shape: assistant{tool_calls, content}, then one tool message per call.
    assert len(out) == 2
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["id"] == "call_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "retrieve"
    # OpenAI requires arguments to be a JSON string, not a dict.
    import json

    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "q",
        "search_mode": "factual",
    }
    assert out[0]["content"] == "Answer [1]"
    assert out[1]["role"] == "tool"
    assert out[1]["tool_call_id"] == "call_1"


class TestResolveResponseLanguage:
    def test_resolve_response_language_explicit_override(self):
        from bibilab.config import AIConfig
        from bibilab.routers.chat import resolve_response_language

        cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="", output_language="zh")
        assert resolve_response_language(cfg, ui_lang="en") == "zh"

    def test_resolve_response_language_ui_fallback(self):
        from bibilab.config import AIConfig
        from bibilab.routers.chat import resolve_response_language

        cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="", output_language="ui")
        assert resolve_response_language(cfg, ui_lang="zh") == "zh"

    def test_resolve_response_language_default_is_ui(self):
        from bibilab.config import AIConfig
        from bibilab.routers.chat import resolve_response_language

        cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="")
        # AIConfig default for output_language is "ui"
        assert resolve_response_language(cfg, ui_lang="en") == "en"


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
                        "name": "retrieve",
                        "arguments": {"query": "q", "search_mode": "factual"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
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
        assert "s1" in registry
        entry = registry["s1"]
        assert entry.index == 1
        assert entry.source_id == "s1"
        assert entry.title == "Video One"
        assert "v1_0_10" in entry.chunk_ids

    def test_reseed_accumulates_chunk_ids_for_same_source(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "Answer",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "retrieve",
                        "arguments": {"query": "q", "search_mode": "factual"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "citation_index": 1,
                                    "chunk_id": "v1_0_10",
                                    "video_title": "V1",
                                },
                                {
                                    "source_id": "s1",
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
        entry = registry["s1"]
        assert "v1_0_10" in entry.chunk_ids
        assert "v1_10_20" in entry.chunk_ids
        assert len(entry.chunk_ids) == 2

    def test_reseed_skips_non_retrieve_blocks(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        history = [
            {
                "role": "assistant",
                "content": "There are 5 videos.",
                "tool_blocks": [
                    {
                        "tool_use_id": "t1",
                        "name": "query_list_metadata",
                        "arguments": {"query_type": "count"},
                        "result": {"count": 5},
                    }
                ],
            }
        ]

        reseed_citation_registry(registry, history)

        assert len(registry) == 0

    def test_reseed_preserves_existing_entries(self):
        from bibilab.pipeline.chat_tools import CitationRegistryEntry, reseed_citation_registry

        registry = {
            "s1": CitationRegistryEntry(index=1, source_id="s1", title="V1", chunk_ids={"v1_0_10"}),
        }
        history = [
            {
                "role": "assistant",
                "content": "More info [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "t2",
                        "name": "retrieve",
                        "arguments": {"query": "q2", "search_mode": "factual"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
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
        entry = registry["s1"]
        assert entry.index == 1
        # Old chunk is preserved
        assert "v1_0_10" in entry.chunk_ids
        # New chunk is added
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
                        "name": "retrieve",
                        "arguments": {"query": "q", "search_mode": "factual"},
                        "result": {
                            "chunks": [
                                {
                                    "source_id": "s1",
                                    "chunk_id": "v1_0_10",
                                    "video_title": "V1",
                                    # citation_index intentionally missing
                                },
                                {
                                    "source_id": "s2",
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

        # Chunk without citation_index is skipped; only s2 is registered.
        assert "s1" not in registry
        assert "s2" in registry
        assert registry["s2"].index == 2

    def test_reseed_empty_history(self):
        from bibilab.pipeline.chat_tools import reseed_citation_registry

        registry: dict = {}
        reseed_citation_registry(registry, [])
        assert len(registry) == 0

        # Also verify an existing entry is preserved
        from bibilab.pipeline.chat_tools import CitationRegistryEntry

        registry_with_entry = {
            "s1": CitationRegistryEntry(index=1, source_id="s1", title="V1"),
        }
        reseed_citation_registry(registry_with_entry, [])
        assert len(registry_with_entry) == 1
        assert "s1" in registry_with_entry
