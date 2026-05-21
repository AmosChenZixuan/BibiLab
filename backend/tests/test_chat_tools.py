"""Tests for generate_report tool execution."""

import logging
from unittest.mock import AsyncMock, patch

import pytest


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

    def test_tool_schema_excludes_titles(self):
        from bibilab.pipeline.chat_tools import QUERY_LIST_METADATA_TOOL

        props = QUERY_LIST_METADATA_TOOL.parameters["properties"]
        assert "titles" not in props["query_type"]["enum"]


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

    @pytest.mark.asyncio
    async def test_unknown_query_type_warns_for_titles(self, tmp_bibilab_home, caplog):
        """After 'titles' is removed from the enum, requesting it falls back to count."""
        import logging
        from unittest.mock import AsyncMock, patch

        from bibilab.pipeline.chat_tools import execute_query_list_metadata

        with patch("bibilab.pipeline.chat_tools.count_sources", new_callable=AsyncMock) as m:
            m.return_value = 3
            with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
                result = await execute_query_list_metadata(["s1"], "titles")

        assert result == {"count": 3}
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
            source_ids=["s1"],
            cfg=None,
        )

        assert result["query"] == "面食 种类"
        assert "filter_miss" not in result  # filter_miss removed in #287


@pytest.mark.asyncio
async def test_execute_retrieve_returns_raw_chunks_for_replay(monkeypatch):
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, params, **kwargs):
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


# AC1: mode is present in execute_retrieve result
@pytest.mark.asyncio
async def test_execute_retrieve_includes_mode_in_result(monkeypatch):
    """execute_retrieve result dict must include mode key."""
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="test",
                    video_title="V",
                    timestamp_start=0.0,
                    timestamp_end=10.0,
                    video_id="v1",
                    distance=0.0,
                    score=0.9,
                ),
            ],
            candidates_evaluated=1,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(video_id="v1", video_title="V", best_score=-0.9)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    result = await chat_tools.execute_retrieve(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry={},
        source_map={"v1": "s1"},
        mode="few",
    )

    assert "mode" in result
    assert result["mode"] == "few"


# AC1: default mode is "narrow"
@pytest.mark.asyncio
async def test_execute_retrieve_default_mode_is_narrow(monkeypatch):
    """When mode is not provided, defaults to 'narrow'."""
    from bibilab.config import AIConfig, BibilabConfig
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

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    result = await chat_tools.execute_retrieve(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
    )

    assert result["mode"] == "narrow"


# AC2: CitationRegistryEntry gets timestamp_start, timestamp_end, rerank_score, preview
@pytest.mark.asyncio
async def test_citation_registry_entry_gets_chunk_fields_on_execute_retrieve(monkeypatch):
    """When chunks are registered, CitationRegistryEntry gets timestamp_start, timestamp_end, rerank_score, preview."""
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="test content here",
                    video_title="Test Video",
                    timestamp_start=120.4,
                    timestamp_end=145.0,
                    video_id="v1",
                    distance=0.0,
                    score=0.95,
                ),
            ],
            candidates_evaluated=1,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(video_id="v1", video_title="Test Video", best_score=-0.95)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    registry: dict = {}
    source_map = {"v1": "s1"}

    await chat_tools.execute_retrieve(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
        source_map=source_map,
    )

    assert "s1" in registry
    entry = registry["s1"]
    assert entry.timestamp_start == 120.4, f"expected 120.4, got {entry.timestamp_start}"
    assert entry.timestamp_end == 145.0, f"expected 145.0, got {entry.timestamp_end}"
    assert entry.rerank_score == 0.95, f"expected 0.95, got {entry.rerank_score}"
    assert entry.preview == "test content here", f"expected stripped preview, got {entry.preview}"


# AC2: multiple chunks from same source only populate first chunk's fields
@pytest.mark.asyncio
async def test_citation_registry_entry_uses_first_chunk_fields(monkeypatch):
    """When multiple chunks come from the same source, only first chunk's fields are stored on the entry."""
    from bibilab.config import AIConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="first chunk",
                    video_title="V",
                    timestamp_start=10.0,
                    timestamp_end=20.0,
                    video_id="v1",
                    distance=0.0,
                    score=0.8,
                ),
                RetrievedChunk(
                    content="second chunk",
                    video_title="V",
                    timestamp_start=30.0,
                    timestamp_end=40.0,
                    video_id="v1",
                    distance=0.0,
                    score=0.9,
                ),
            ],
            candidates_evaluated=2,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(video_id="v1", video_title="V", best_score=-0.9)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
    registry: dict = {}
    source_map = {"v1": "s1"}

    await chat_tools.execute_retrieve(
        query="test",
        source_ids=["s1"],
        cfg=cfg,
        registry=registry,
        source_map=source_map,
    )

    entry = registry["s1"]
    # First chunk's timestamps, not second
    assert entry.timestamp_start == 10.0
    assert entry.timestamp_end == 20.0
    # First chunk's score
    assert entry.rerank_score == 0.8


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

    def test_build_grounding_prompt_localizes_style_directive(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="zh")
        # Style directive still interpolates the response language.
        assert "Answer in zh" in prompt
        # And the old hard-coded English string must be gone.
        assert "The provided sources do not cover this topic" not in prompt

    def test_build_grounding_prompt_no_retrieve_tool(self):
        """Answer LLM no longer has retrieve — prompt must frame excerpts as
        pre-retrieved, not ask for new retrieval. Refusal only from completed
        retrieve result (no pre-retrieve shortcut).
        """
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="zh")
        # No retrieve instruction — answer LLM doesn't have the tool.
        assert "call `retrieve`" not in prompt
        assert "do not call `retrieve`" not in prompt
        # Pre-retrieve refusal shortcut absent.
        assert "say so in zh" not in prompt
        assert "say so in en" not in build_grounding_prompt(response_language="en")
        # Outside-knowledge / analogy ban present.
        assert "outside knowledge" in prompt
        assert "real-world analogies" in prompt
        # Mentions pre-retrieved excerpts.
        assert "have been retrieved" in prompt
        assert "tool results" in prompt


class TestBuildToolBlockEntry:
    def test_build_tool_block_entry_retrieve_strips_internal_underscore_fields(self):
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        retrieve_result = {
            "query": "test",
            "mode": "narrow",
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
            arguments={"query": "test", "mode": "narrow"},
            result=retrieve_result,
            raw_chunks=raw_chunks,
        )

        assert entry["tool_use_id"] == "toolu_1"
        assert entry["name"] == "retrieve"
        assert entry["arguments"] == {"query": "test", "mode": "narrow"}
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
                "arguments": {"query": "q", "expected_hits": "few"},
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
        "input": {"query": "q", "expected_hits": "few"},
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
                "arguments": {"query": "q", "expected_hits": "few"},
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
        "expected_hits": "few",
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


class TestFacetInt:
    """#309: _facet_int wraps the shared parse_facet_int, degrading unusable LLM
    args to None (never raises). Coercion rules (>=1, bool/non-integral rejected)
    are single-sourced in parse_facet_int — see digest.py."""

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (8, 8),
            ("8", 8),
            (8.0, 8),
            (" 8 ", 8),
            (1, 1),
            (None, None),
            # < 1 rejected by parse_facet_int (every stored facet is >= 1) → drop
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
        """M2: a nonsensical episode (0/negative/garbage) is logged + dropped,
        not silently turned into a guaranteed zero-match."""
        from bibilab.pipeline.chat_tools import _facet_int

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            assert _facet_int(0, "sequence_number") is None
        assert "unusable, dropping predicate" in caplog.text


class TestExecuteRetrieveFacetScoping:
    """#309: deterministic facet scoping with fail-open."""

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

        async def fake_retrieve(query_text, source_ids, cfg, params, scoped_source_ids=None, **kw):
            captured["scoped"] = scoped_source_ids
            return self._empty_result()

        async def fake_get_source_facets(ids):
            return {k: v for k, v in facets.items() if k in ids}

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_tools, "get_source_facets", fake_get_source_facets)
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
        result = await chat_tools.execute_retrieve(
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
        # Decision B: scoped_pool_size stays PRE-facet (scope none → len(source_ids)=3),
        # while facet_scope.matched_count carries the narrowed count (2).
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
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=99,
        )
        assert captured["scoped"] is None  # pre-facet pool was None (scope none) — unchanged
        assert result["facet_scope"]["no_match"] is True
        assert result["facet_scope"]["matched_count"] == 0

    @pytest.mark.asyncio
    async def test_no_facet_params_byte_identical(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(monkeypatch, {})
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
        )
        assert captured["scoped"] is None  # regression guard: unchanged from pre-#309
        assert result["facet_scope"] == {
            "sequence_number": None,
            "season_number": None,
            "matched_count": None,
            "no_match": False,
        }

    @pytest.mark.asyncio
    async def test_no_match_prepends_note_to_llm_chunks(self, monkeypatch):
        """#310: facet_scope.no_match true ⇒ _chunks carries the LLM-visible note."""
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_MATCH_NOTE

        self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": None},
                "b": {"sequence_number": 2, "season_number": None},
            },
        )
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=99,  # no source has seq 99 → no_match
        )
        assert result["facet_scope"]["no_match"] is True
        # _NO_MATCH_NOTE present (coexists with _NO_COVERAGE_NOTE when retrieve
        # also returned zero chunks — the empty-result mock always does so).
        assert _NO_MATCH_NOTE in result["_chunks"]

    @pytest.mark.asyncio
    async def test_match_does_not_prepend_note(self, monkeypatch):
        """#310: matched (or no-facet) path leaves _chunks free of the note."""
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_MATCH_NOTE

        self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": None},
                "b": {"sequence_number": 2, "season_number": None},
            },
        )
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
        )
        assert result["facet_scope"]["no_match"] is False
        assert _NO_MATCH_NOTE not in result["_chunks"]

    @pytest.mark.asyncio
    async def test_season_omitted_does_not_filter(self, monkeypatch):
        """Both sources share sequence_number but differ in season; season is
        omitted from the call → season must NOT narrow (both kept). Distinguishes
        'season ignored' from 'season coincidentally shared' (the latter is false
        here: seasons are 2 vs 5)."""
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": 2},
                "b": {"sequence_number": 1, "season_number": 5},
            },
        )
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
        )
        assert sorted(captured["scoped"]) == ["a", "b"]
        assert result["facet_scope"]["matched_count"] == 2

    @pytest.mark.asyncio
    async def test_season_predicate_and_two_key_intersection(self, monkeypatch):
        """season_number IS passed and must AND with sequence_number. 'a' and 'b'
        share sequence_number=1; only 'a' has season_number=2. Guards: all→any
        regression (would wrongly keep 'b'), and dropping season_number from the
        predicate dict (would wrongly keep 'b')."""
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "a": {"sequence_number": 1, "season_number": 2},
                "b": {"sequence_number": 1, "season_number": 5},
            },
        )
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["a", "b"],
            cfg=self._cfg(),
            sequence_number=1,
            season_number=2,
        )
        assert captured["scoped"] == ["a"]  # b excluded: season 5 != 2
        assert result["facet_scope"] == {
            "sequence_number": 1,
            "season_number": 2,
            "matched_count": 1,
            "no_match": False,
        }


class TestExecuteRetrieveZeroChunkNote:
    """Refusal must be anchored to evidence: zero chunks above the relevance
    gate ⇒ _NO_COVERAGE_NOTE prepended to _chunks so the LLM follows a
    deterministic refusal instruction instead of inventing one pre-retrieve."""

    @staticmethod
    def _cfg():
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig

        return BibilabConfig(
            ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
            backend=BackendConfig(),
        )

    @pytest.mark.asyncio
    async def test_zero_chunks_prepends_no_coverage_note(self, monkeypatch):
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_COVERAGE_NOTE
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(query_text, source_ids, cfg, params, **kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=42,
                sources_with_hits=0,
                sources_total=3,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        result = await chat_tools.execute_retrieve(
            query="什么是多元主义",
            source_ids=["s1", "s2", "s3"],
            cfg=self._cfg(),
        )

        assert result["_chunks"].startswith(_NO_COVERAGE_NOTE)

    @pytest.mark.asyncio
    async def test_non_zero_chunks_omits_no_coverage_note(self, monkeypatch):
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.chat_tools import _NO_COVERAGE_NOTE
        from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

        async def fake_retrieve(query_text, source_ids, cfg, params, **kwargs):
            return RetrievalResult(
                chunks=[
                    RetrievedChunk(
                        content="hit",
                        video_title="V1",
                        timestamp_start=0.0,
                        timestamp_end=5.0,
                        video_id="v1",
                        distance=0.1,
                        score=0.9,
                    ),
                ],
                source_coverage=[SourceHit(video_id="v1", video_title="V1", best_score=-0.9)],
                candidates_evaluated=1,
                sources_with_hits=1,
                sources_total=1,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["s1"],
            cfg=self._cfg(),
            source_map={"v1": "s1"},
        )

        assert _NO_COVERAGE_NOTE not in result["_chunks"]


def test_build_fenced_chunks_groups_and_fences():
    from bibilab.pipeline.chat_tools import (
        CitationRegistryEntry,
        _build_fenced_chunks,
    )

    registry = {
        "s1": CitationRegistryEntry(index=1, source_id="s1", title="Video One"),
        "s2": CitationRegistryEntry(index=2, source_id="s2", title="Video Two"),
    }
    # Insertion order deliberately interleaved + index 2 before 1 to prove
    # the helper groups by index and emits ascending, not insertion order.
    chunks_by_index = {
        2: ['[2 @ 0s-9s]: "b1"', '[2 @ 9s-18s]: "b2"'],
        1: ['[1 @ 0s-9s]: "a1"', '[1 @ 9s-18s]: "a2"'],
    }

    out = _build_fenced_chunks(chunks_by_index, registry)

    assert out == (
        '===== Source [1]: "Video One" =====\n'
        '[1 @ 0s-9s]: "a1"\n'
        '[1 @ 9s-18s]: "a2"\n'
        "\n"
        '===== Source [2]: "Video Two" =====\n'
        '[2 @ 0s-9s]: "b1"\n'
        '[2 @ 9s-18s]: "b2"'
    )


def test_build_fenced_chunks_empty_returns_empty_string():
    from bibilab.pipeline.chat_tools import _build_fenced_chunks

    assert _build_fenced_chunks({}, {}) == ""


@pytest.mark.asyncio
async def test_execute_retrieve_chunks_grouped_and_fenced(monkeypatch):
    """#297: _chunks groups by source + fences each group; _raw_chunks
    preserves the original interleaved rerank order (invariant)."""
    import json

    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.chat_tools import build_tool_block_entry
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

    async def fake_retrieve(query_text, source_ids, cfg, params, **kwargs):
        # Rerank-interleaved order: v1, v2, v1, v2.
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="a1",
                    video_title="Video One",
                    timestamp_start=0.0,
                    timestamp_end=9.0,
                    video_id="v1",
                    distance=0.1,
                    score=0.9,
                ),
                RetrievedChunk(
                    content="b1",
                    video_title="Video Two",
                    timestamp_start=0.0,
                    timestamp_end=9.0,
                    video_id="v2",
                    distance=0.1,
                    score=0.8,
                ),
                RetrievedChunk(
                    content="a2",
                    video_title="Video One",
                    timestamp_start=9.0,
                    timestamp_end=18.0,
                    video_id="v1",
                    distance=0.2,
                    score=0.7,
                ),
                RetrievedChunk(
                    content="b2",
                    video_title="Video Two",
                    timestamp_start=9.0,
                    timestamp_end=18.0,
                    video_id="v2",
                    distance=0.2,
                    score=0.6,
                ),
            ],
            candidates_evaluated=4,
            sources_with_hits=2,
            sources_total=2,
            source_coverage=[
                SourceHit(video_id="v1", video_title="Video One", best_score=-0.9),
                SourceHit(video_id="v2", video_title="Video Two", best_score=-0.8),
            ],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )
    registry: dict = {}
    source_map = {"v1": "s1", "v2": "s2"}

    result = await chat_tools.execute_retrieve(
        query="q",
        source_ids=["s1", "s2"],
        cfg=cfg,
        registry=registry,
        source_map=source_map,
    )

    chunks = result["_chunks"]

    # Fences present, both sources, ascending order.
    f1 = chunks.index('===== Source [1]: "Video One" =====')
    f2 = chunks.index('===== Source [2]: "Video Two" =====')
    assert f1 < f2

    # Grouping: both v1 chunk lines precede any v2 chunk line.
    a2 = chunks.index('[1 @ 9s-18s]: "a2"')
    b1 = chunks.index('[2 @ 0s-9s]: "b1"')
    assert a2 < b1, "v1 chunks must cluster before v2 chunks (no interleave)"

    # Cumulative _build_source_headers anchor still present (non-fenced line).
    assert 'Source [1]: "Video One"' in chunks
    assert "Sources retrieved this turn:" in chunks

    # INVARIANT: _raw_chunks keeps original interleaved order + indices.
    raw = result["_raw_chunks"]
    assert [r["citation_index"] for r in raw] == [1, 2, 1, 2]
    assert [r["content"] for r in raw] == ["a1", "b1", "a2", "b2"]

    # INVARIANT: fences must not leak into the persisted replay block.
    entry = build_tool_block_entry("tu1", "retrieve", {}, result, result["_raw_chunks"])
    assert "=====" not in json.dumps(entry, ensure_ascii=False)
    assert "_chunks" not in entry["result"]


def test_retrieve_tool_definition_deleted():
    import bibilab.pipeline.chat_tools as ct

    assert not hasattr(ct, "RETRIEVE_TOOL")
    assert not hasattr(ct, "_PARAMS_BY_HITS")
    assert not hasattr(ct, "params_for_expected_hits")
    assert not hasattr(ct, "DEFAULT_EXPECTED_HITS")


def test_trivial_ack_helpers_deleted():
    import bibilab.pipeline.chat_tools as ct

    assert not hasattr(ct, "trivial_ack_note")
    assert not hasattr(ct, "_TRIVIAL_STOPWORDS")
    assert not hasattr(ct, "_TRIVIAL_PATH_NOTE")


def test_rewriter_mode_to_params_mapping():
    from bibilab.pipeline.chat_tools import _REWRITER_MODE_TO_PARAMS

    narrow = _REWRITER_MODE_TO_PARAMS["narrow"]
    assert (narrow.depth_per_source, narrow.top_k, narrow.mode) == (2, 8, "narrow")

    survey = _REWRITER_MODE_TO_PARAMS["survey"]
    assert (survey.depth_per_source, survey.top_k, survey.mode) == (5, 24, "survey")


def test_retrieval_params_uses_mode_field():
    from bibilab.models._enums import _RELEVANCE_MARGIN_BY_MODE, RetrievalParams

    p = RetrievalParams(depth_per_source=2, top_k=8, mode="narrow")
    assert p.mode == "narrow"
    assert _RELEVANCE_MARGIN_BY_MODE["narrow"] == 2.0
    assert _RELEVANCE_MARGIN_BY_MODE["survey"] == 2.5


def test_expected_hits_symbols_deleted():
    import bibilab.models._enums as e

    assert not hasattr(e, "ExpectedHits")
    assert not hasattr(e, "_RELEVANCE_MARGIN_BY_HITS")
