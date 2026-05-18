"""Tests for generate_report tool execution."""

import logging
from unittest.mock import AsyncMock, patch

import pytest


class TestRetrieveToolSchema:
    def test_retrieve_tool_no_search_mode(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "search_mode" not in props

    def test_retrieve_tool_no_source_filter(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "source_filter" not in props

    def test_retrieve_tool_has_no_index_scope_params(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "source_ids" not in props
        assert "exclude_source_ids" not in props

    def test_retrieve_tool_has_expected_hits(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "expected_hits" in props
        assert props["expected_hits"]["enum"] == ["one", "few", "many"]

    def test_retrieve_tool_required_is_query_only(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        assert RETRIEVE_TOOL.parameters["required"] == ["query"]

    def test_retrieve_tool_description_has_no_index_workflow(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        assert "exclude_source_ids" not in desc
        assert "source numbers" not in desc
        assert "Sources list" not in desc
        assert "sequence_number" in desc


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
    async def test_execute_retrieve_with_selected_source_ids_limits_search_pool(self, monkeypatch):
        """selected_source_ids set intersection should limit search pool."""
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

        retrieve_called_with = None

        async def fake_retrieve(query_text, source_ids, cfg, params, scoped_source_ids=None, **kwargs):
            nonlocal retrieve_called_with
            retrieve_called_with = scoped_source_ids
            return RetrievalResult(
                chunks=[
                    RetrievedChunk(
                        content="content about s2",
                        video_title="Source 2",
                        timestamp_start=0.0,
                        timestamp_end=10.0,
                        video_id="v2",
                        distance=0.0,
                    ),
                ],
                candidates_evaluated=1,
                sources_with_hits=1,
                sources_total=3,
                source_coverage=[SourceHit(video_id="v2", video_title="Source 2", best_score=0.0)],
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test query",
            source_ids=["s1", "s2", "s3"],
            cfg=cfg,
            source_map={"v2": "s2"},
            selected_source_ids=["s2"],
            scope_choice="whitelist",
        )

        assert retrieve_called_with == ["s2"]
        assert result["sources_total"] == 3

    @pytest.mark.asyncio
    async def test_execute_retrieve_empty_selected_source_ids_means_search_all(self, monkeypatch):
        """selected_source_ids=[] is falsy, scoped_source_ids=None (search all)."""
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        retrieve_called_with = None

        async def fake_retrieve(query_text, source_ids, cfg, params, scoped_source_ids=None, **kwargs):
            nonlocal retrieve_called_with
            retrieve_called_with = scoped_source_ids
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=2,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2"],
            cfg=cfg,
            selected_source_ids=[],  # empty list means "search all"
        )

        assert retrieve_called_with is None  # must be None, not [], to mean "search all"


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


# AC1: expected_hits is present in execute_retrieve result
@pytest.mark.asyncio
async def test_execute_retrieve_includes_expected_hits_in_result(monkeypatch):
    """execute_retrieve result dict must include expected_hits key."""
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
        expected_hits="few",
    )

    assert "expected_hits" in result
    assert result["expected_hits"] == "few"


# AC1: default expected_hits is "few"
@pytest.mark.asyncio
async def test_execute_retrieve_default_expected_hits_is_few(monkeypatch):
    """When expected_hits is not provided, defaults to 'few'."""
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

    assert result["expected_hits"] == "few"


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

    def test_build_grounding_prompt_localizes_fallback_sentence(self):
        from bibilab.routers.chat import build_grounding_prompt

        prompt = build_grounding_prompt(response_language="zh")
        # The fallback rule must reference the response_language, not hard-coded English.
        assert "say so in zh" in prompt
        assert "do not call `retrieve` again" in prompt
        # And the old hard-coded English string must be gone.
        assert "The provided sources do not cover this topic" not in prompt

    # (fresh retrieve directive removed — now handled by build_grounding_prompt reuse
    #  instruction + retrieve tool description)
    # (verbatim proper noun rule removed — replaced by ## Grounding "Copy proper nouns ... verbatim")
    # (no real-world parallels rule removed — covered by fiction-authoritative sentence in ## Grounding)


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
        assert "source_ids" in desc
        assert "expected_hits" in desc


class TestBuildToolBlockEntry:
    def test_build_tool_block_entry_retrieve_strips_internal_underscore_fields(self):
        from bibilab.pipeline.chat_tools import build_tool_block_entry

        retrieve_result = {
            "query": "test",
            "expected_hits": "few",
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
            arguments={"query": "test", "expected_hits": "few"},
            result=retrieve_result,
            raw_chunks=raw_chunks,
        )

        assert entry["tool_use_id"] == "toolu_1"
        assert entry["name"] == "retrieve"
        assert entry["arguments"] == {"query": "test", "expected_hits": "few"}
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


# =============================================================================
# Tests for I-6a: retrieve scope semantics — blacklist/exclude
# =============================================================================


class TestCoerceToStrList:
    """AC2: _coerce_to_str_list parses JSON-string lists without silently dropping scope."""

    def test_coerce_list_of_strings(self):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        result = _coerce_to_str_list(["1", "3", "7"], "exclude_source_ids")
        assert result == ["1", "3", "7"]

    def test_coerce_stringified_json_list(self):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        result = _coerce_to_str_list('["1", "3", "7"]', "exclude_source_ids")
        assert result == ["1", "3", "7"]

    def test_coerce_none(self):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        result = _coerce_to_str_list(None, "exclude_source_ids")
        assert result is None

    def test_coerce_garbage_string_warns(self, caplog):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            result = _coerce_to_str_list("not a list at all", "exclude_source_ids")

        assert result is None
        assert "unparseable" in caplog.text

    def test_coerce_mixed_type_list(self):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        result = _coerce_to_str_list([1, "2", 3], "source_ids")
        assert result == ["1", "2", "3"]

    def test_coerce_non_scalar_items_skipped(self, caplog):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            result = _coerce_to_str_list([["1"], "2"], "exclude_source_ids")

        assert result == ["2"]
        assert "non-scalar" in caplog.text

    def test_coerce_integer_returns_none(self, caplog):
        from bibilab.pipeline.chat_tools import _coerce_to_str_list

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            result = _coerce_to_str_list(42, "exclude_source_ids")

        assert result is None
        assert "unparseable" in caplog.text


class TestMapIndicesToUuids:
    """AC6: _map_indices_to_uuids extracts index-or-UUID mapping logic."""

    def test_maps_1_based_indices_to_uuids(self):
        from bibilab.pipeline.chat_tools import _map_indices_to_uuids

        result = _map_indices_to_uuids(["1", "3"], ["uuid-a", "uuid-b", "uuid-c"])
        assert result == ["uuid-a", "uuid-c"]

    def test_accepts_raw_uuids(self):
        from bibilab.pipeline.chat_tools import _map_indices_to_uuids

        result = _map_indices_to_uuids(["uuid-a", "uuid-c"], ["uuid-a", "uuid-b", "uuid-c"])
        assert result == ["uuid-a", "uuid-c"]

    def test_out_of_range_index_logs_warning(self, caplog):
        from bibilab.pipeline.chat_tools import _map_indices_to_uuids

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            result = _map_indices_to_uuids(["1", "999"], ["uuid-a", "uuid-b"])

        assert result == ["uuid-a"]
        assert "out of range" in caplog.text

    def test_unrecognized_identifier_logs_warning(self, caplog):
        from bibilab.pipeline.chat_tools import _map_indices_to_uuids

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            result = _map_indices_to_uuids(["uuid-a", "garbage"], ["uuid-a", "uuid-b"])

        assert result == ["uuid-a"]
        assert "Unrecognized source identifier" in caplog.text


class TestExecuteToolDispatchMatrix:
    """AC1: execute_tool dispatch uses exclude when both exclude and whitelist are present."""

    @pytest.mark.asyncio
    async def test_exclude_only_uses_exclude_path(self, tmp_bibilab_home):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        async def patched_retrieve(*args, **kwargs):
            return await fake_retrieve(*args, **kwargs)

        with patch.object(chat_tools, "retrieve", side_effect=patched_retrieve):
            result = await chat_tools.execute_tool(
                tool_name="retrieve",
                arguments={
                    "query": "test query",
                    "exclude_source_ids": ["1", "2"],
                },
                list_id="list-1",
                source_ids=["s1", "s2", "s3"],
                ui_lang="en",
                cfg=cfg,
            )

        assert result["scope_choice"] == "exclude"
        assert result["excluded_count"] == 2
        assert result["scoped_pool_size"] == 1

    @pytest.mark.asyncio
    async def test_whitelist_only_uses_whitelist_path(self, tmp_bibilab_home):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        async def patched_retrieve(*args, **kwargs):
            return await fake_retrieve(*args, **kwargs)

        with patch.object(chat_tools, "retrieve", side_effect=patched_retrieve):
            result = await chat_tools.execute_tool(
                tool_name="retrieve",
                arguments={
                    "query": "test query",
                    "source_ids": ["1"],
                },
                list_id="list-1",
                source_ids=["s1", "s2", "s3"],
                ui_lang="en",
                cfg=cfg,
            )

        assert result["scope_choice"] == "whitelist"
        assert result["excluded_count"] is None
        assert result["scoped_pool_size"] == 1

    @pytest.mark.asyncio
    async def test_both_given_uses_exclude_with_warning(self, tmp_bibilab_home, caplog):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        async def patched_retrieve(*args, **kwargs):
            return await fake_retrieve(*args, **kwargs)

        with patch.object(chat_tools, "retrieve", side_effect=patched_retrieve):
            with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
                result = await chat_tools.execute_tool(
                    tool_name="retrieve",
                    arguments={
                        "query": "test query",
                        "exclude_source_ids": ["1"],
                        "source_ids": ["2"],
                    },
                    list_id="list-1",
                    source_ids=["s1", "s2", "s3"],
                    ui_lang="en",
                    cfg=cfg,
                )

        assert result["scope_choice"] == "exclude"
        assert "both" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_neither_given_uses_none_with_warning(self, tmp_bibilab_home, caplog):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        async def patched_retrieve(*args, **kwargs):
            return await fake_retrieve(*args, **kwargs)

        with patch.object(chat_tools, "retrieve", side_effect=patched_retrieve):
            with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
                result = await chat_tools.execute_tool(
                    tool_name="retrieve",
                    arguments={
                        "query": "test query",
                    },
                    list_id="list-1",
                    source_ids=["s1", "s2", "s3"],
                    ui_lang="en",
                    cfg=cfg,
                )

        assert result["scope_choice"] == "none"
        assert "neither" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_stringified_json_list_coerced_successfully(self, tmp_bibilab_home):
        """AC2: JSON string passed as exclude_source_ids is coerced to list."""
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        async def patched_retrieve(*args, **kwargs):
            return await fake_retrieve(*args, **kwargs)

        with patch.object(chat_tools, "retrieve", side_effect=patched_retrieve):
            result = await chat_tools.execute_tool(
                tool_name="retrieve",
                arguments={
                    "query": "test query",
                    "exclude_source_ids": '["1", "2"]',  # string, not list
                },
                list_id="list-1",
                source_ids=["s1", "s2", "s3"],
                ui_lang="en",
                cfg=cfg,
            )

        assert result["scope_choice"] == "exclude"
        assert result["excluded_count"] == 2
        assert result["scoped_pool_size"] == 1


class TestExecuteRetrieveExcludePath:
    """AC6: execute_retrieve exclude path maps indices to correct UUIDs; out-of-range index produces warning."""

    @pytest.mark.asyncio
    async def test_exclude_indices_map_to_correct_uuids(self, monkeypatch):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        retrieve_called_with = None

        async def fake_retrieve(query_text, source_ids, cfg, params, scoped_source_ids=None, **kwargs):
            nonlocal retrieve_called_with
            retrieve_called_with = scoped_source_ids
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        await chat_tools.execute_retrieve(
            query="test",
            source_ids=["uuid-a", "uuid-b", "uuid-c"],
            cfg=cfg,
            exclude_source_ids=["1", "3"],
            scope_choice="exclude",
        )

        assert retrieve_called_with == ["uuid-b"]  # excluded: uuid-a, uuid-c

    @pytest.mark.asyncio
    async def test_exclude_out_of_range_index_logs_warning(self, monkeypatch, caplog):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.chat_tools"):
            await chat_tools.execute_retrieve(
                query="test",
                source_ids=["uuid-a", "uuid-b"],
                cfg=cfg,
                exclude_source_ids=["1", "999"],
                scope_choice="exclude",
            )

        assert "out of range" in caplog.text


class TestExecuteRetrieveTelemetry:
    """AC3: execute_retrieve result includes scope_choice, excluded_count, scoped_pool_size."""

    @pytest.mark.asyncio
    async def test_exclude_path_includes_telemetry(self, monkeypatch):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[
                    RetrievedChunk(
                        content="content",
                        video_title="V2",
                        timestamp_start=0.0,
                        timestamp_end=10.0,
                        video_id="v2",
                        distance=0.0,
                        score=0.9,
                    ),
                ],
                candidates_evaluated=1,
                sources_with_hits=1,
                sources_total=3,
                source_coverage=[SourceHit(video_id="v2", video_title="V2", best_score=-0.9)],
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2", "s3"],
            cfg=cfg,
            source_map={"v2": "s2"},
            exclude_source_ids=["1"],
            scope_choice="exclude",
        )

        assert "scope_choice" in result
        assert "excluded_count" in result
        assert "scoped_pool_size" in result
        assert result["scope_choice"] == "exclude"
        assert result["excluded_count"] == 1
        assert result["scoped_pool_size"] == 2  # 3 - 1 excluded

    @pytest.mark.asyncio
    async def test_whitelist_path_includes_telemetry(self, monkeypatch):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[
                    RetrievedChunk(
                        content="content",
                        video_title="V2",
                        timestamp_start=0.0,
                        timestamp_end=10.0,
                        video_id="v2",
                        distance=0.0,
                        score=0.9,
                    ),
                ],
                candidates_evaluated=1,
                sources_with_hits=1,
                sources_total=3,
                source_coverage=[SourceHit(video_id="v2", video_title="V2", best_score=-0.9)],
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2", "s3"],
            cfg=cfg,
            source_map={"v2": "s2"},
            selected_source_ids=["2"],
            scope_choice="whitelist",
        )

        assert result["scope_choice"] == "whitelist"
        assert result["excluded_count"] is None
        assert result["scoped_pool_size"] == 1

    @pytest.mark.asyncio
    async def test_none_scope_path_includes_telemetry(self, monkeypatch):
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[
                    RetrievedChunk(
                        content="content",
                        video_title="V1",
                        timestamp_start=0.0,
                        timestamp_end=10.0,
                        video_id="v1",
                        distance=0.0,
                        score=0.9,
                    ),
                ],
                candidates_evaluated=1,
                sources_with_hits=1,
                sources_total=3,
                source_coverage=[SourceHit(video_id="v1", video_title="V1", best_score=-0.9)],
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2", "s3"],
            cfg=cfg,
            source_map={"v1": "s1"},
            scope_choice=None,
        )

        assert result["scope_choice"] == "none"
        assert result["excluded_count"] is None
        assert result["scoped_pool_size"] == 3  # all sources

    @pytest.mark.asyncio
    async def test_exclude_empty_list_is_valid(self, monkeypatch):
        """Empty exclude_source_ids=[] means nothing excluded; all sources searched."""
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2", "s3"],
            cfg=cfg,
            exclude_source_ids=[],
            scope_choice="exclude",
        )

        assert result["scope_choice"] == "exclude"
        assert result["excluded_count"] == 0
        assert result["scoped_pool_size"] == 3  # all sources kept

    @pytest.mark.asyncio
    async def test_exclude_all_sources_returns_empty_pool(self, monkeypatch):
        """Excluding all sources produces empty pool, not a crash."""
        from bibilab.config import AIConfig, BibilabConfig
        from bibilab.pipeline import chat_tools
        from bibilab.pipeline.embed import RetrievalResult

        async def fake_retrieve(**kwargs):
            return RetrievalResult(
                chunks=[],
                source_coverage=[],
                candidates_evaluated=0,
                sources_with_hits=0,
                sources_total=3,
            )

        monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)

        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))
        result = await chat_tools.execute_retrieve(
            query="test",
            source_ids=["s1", "s2"],
            cfg=cfg,
            exclude_source_ids=["1", "2"],
            scope_choice="exclude",
        )

        assert result["scope_choice"] == "exclude"
        assert result["excluded_count"] == 2
        assert result["scoped_pool_size"] == 0
        assert result["candidates_evaluated"] == 0


class TestRetrieveToolSchemaUpdate:
    """Verify RETRIEVE_TOOL schema has exclude_source_ids as primary required param."""

    def test_exclude_source_ids_is_required(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        required = RETRIEVE_TOOL.parameters["required"]
        assert "exclude_source_ids" in required
        assert "source_ids" not in required

    def test_exclude_source_ids_in_properties(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "exclude_source_ids" in props
        assert props["exclude_source_ids"]["type"] == "array"
        assert props["exclude_source_ids"]["items"] == {"type": "string"}

    def test_source_ids_retained_as_optional(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert "source_ids" in props
        required = RETRIEVE_TOOL.parameters["required"]
        assert "source_ids" not in required

    def test_retrieve_tool_description_mentions_exclude(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        assert "exclude_source_ids" in desc

    def test_retrieve_tool_description_mentions_whitelist_rare(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        desc = RETRIEVE_TOOL.description
        assert "source_ids" in desc
        assert "whitelist" in desc.lower() or "ONLY when the user explicitly" in desc


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


class TestRetrieveToolFacetSchema:
    """#309: schema exposes optional sequence_number/season_number; no series_name."""

    def test_facet_params_present_and_typed(self):
        from bibilab.pipeline.chat_tools import RETRIEVE_TOOL

        props = RETRIEVE_TOOL.parameters["properties"]
        assert props["sequence_number"]["type"] == "integer"
        assert props["season_number"]["type"] == "integer"
        assert "series_name" not in props
        # facet params are optional
        assert "sequence_number" not in RETRIEVE_TOOL.parameters["required"]
        assert "season_number" not in RETRIEVE_TOOL.parameters["required"]


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
            scope_choice="none",
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
            scope_choice="none",
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
            scope_choice="none",
        )
        assert captured["scoped"] is None  # regression guard: unchanged from pre-#309
        assert result["facet_scope"] == {
            "sequence_number": None,
            "season_number": None,
            "matched_count": None,
            "no_match": False,
        }

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
            scope_choice="none",
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
            scope_choice="none",
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

    @pytest.mark.asyncio
    async def test_facet_intersects_exclude_pool(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "ua": {"sequence_number": 8, "season_number": None},
                "ub": {"sequence_number": 8, "season_number": None},
                "uc": {"sequence_number": 8, "season_number": None},
            },
        )
        # exclude_source_ids=["1"] removes index 1 → "ua"; pre-facet pool = [ub, uc]
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["ua", "ub", "uc"],
            cfg=self._cfg(),
            exclude_source_ids=["1"],
            scope_choice="exclude",
            sequence_number=8,
        )
        assert sorted(captured["scoped"]) == ["ub", "uc"]  # ua excluded, not re-added
        assert result["facet_scope"]["matched_count"] == 2
        # Decision B on the non-trivial path: scoped_pool_size = post-exclude
        # PRE-facet pool ([ub, uc] = 2), NOT the post-facet narrowed count.
        assert result["scoped_pool_size"] == 2

    @pytest.mark.asyncio
    async def test_zero_match_fails_open_to_exclude_pool_not_full(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        captured = self._patch(
            monkeypatch,
            {
                "ua": {"sequence_number": 1, "season_number": None},
                "ub": {"sequence_number": 2, "season_number": None},
                "uc": {"sequence_number": 3, "season_number": None},
            },
        )
        result = await chat_tools.execute_retrieve(
            query="q",
            source_ids=["ua", "ub", "uc"],
            cfg=self._cfg(),
            exclude_source_ids=["1"],
            scope_choice="exclude",
            sequence_number=99,
        )
        # zero facet match → fall back to PRE-FACET pool (post-exclude), NOT full source_ids
        assert sorted(captured["scoped"]) == ["ub", "uc"]
        assert result["facet_scope"]["no_match"] is True


class TestExecuteToolFacetArgs:
    """#309: execute_tool parses + coerces facet args from raw LLM arguments."""

    @staticmethod
    def _cfg():
        from bibilab.config import AIConfig, BibilabConfig

        return BibilabConfig(ai=AIConfig(protocol="openai", model="x", api_key="k"))

    @pytest.mark.asyncio
    async def test_string_facet_arg_coerced_and_forwarded(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        seen = {}

        async def fake_execute_retrieve(**kwargs):
            seen.update(kwargs)
            return {"facet_scope": {}, "_chunks": "", "_turn_indices": [], "_raw_chunks": []}

        monkeypatch.setattr(chat_tools, "execute_retrieve", fake_execute_retrieve)

        await chat_tools.execute_tool(
            tool_name="retrieve",
            arguments={"query": "第八集", "exclude_source_ids": [], "sequence_number": "8"},
            list_id="l1",
            source_ids=["a"],
            ui_lang="zh",
            cfg=self._cfg(),
        )
        assert seen["sequence_number"] == 8  # "8" → 8
        assert seen["season_number"] is None  # absent → None

    @pytest.mark.asyncio
    async def test_non_numeric_facet_arg_dropped(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        seen = {}

        async def fake_execute_retrieve(**kwargs):
            seen.update(kwargs)
            return {"facet_scope": {}, "_chunks": "", "_turn_indices": [], "_raw_chunks": []}

        monkeypatch.setattr(chat_tools, "execute_retrieve", fake_execute_retrieve)

        await chat_tools.execute_tool(
            tool_name="retrieve",
            arguments={"query": "q", "exclude_source_ids": [], "sequence_number": "eight"},
            list_id="l1",
            source_ids=["a"],
            ui_lang="zh",
            cfg=self._cfg(),
        )
        assert seen["sequence_number"] is None  # "eight" → dropped
