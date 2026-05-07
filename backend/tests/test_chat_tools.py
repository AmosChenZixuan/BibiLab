"""Tests for generate_report tool execution."""

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
