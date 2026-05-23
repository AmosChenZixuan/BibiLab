"""Tests for CitationRegistryEntry behavior across multiple retrieve calls."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from bibilab.config import AIConfig, BibilabConfig
from bibilab.pipeline.chat_tools import CitationRegistryEntry, execute_retrieve


@dataclass
class FakeSourceHit:
    video_id: str
    video_title: str
    best_score: float = -1.0


@dataclass
class FakeRetrievedChunk:
    content: str
    video_title: str
    timestamp_start: float
    timestamp_end: float
    video_id: str
    distance: float = 0.0
    score: float | None = None


def _make_result(source_hits, chunks):
    """Build a RetrievalResult-shaped mock."""
    result = AsyncMock()
    result.source_coverage = source_hits
    result.chunks = chunks
    result.candidates_evaluated = len(chunks)
    result.sources_with_hits = len(source_hits)
    result.sources_total = len(source_hits)
    return result


class TestFirstRetrieveAssignsIndicesStartingAtOne:
    @pytest.mark.asyncio
    async def test_first_retrieve_assigns_indices_starting_at_1(self, tmp_bibilab_home):
        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v1", video_title="Video One"),
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                ],
                [
                    FakeRetrievedChunk(
                        content="chunk",
                        video_title="Video One",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v1",
                    )
                ],
            )
            registry = {}
            source_map = {"v1": "s1", "v2": "s2"}

            result = await execute_retrieve(
                query="test",
                tool_name="retrieve",
                source_ids=["s1", "s2"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )

        assert registry["s1"].index == 1
        assert registry["s2"].index == 2
        assert result["_turn_indices"] == [1, 2]


class TestSecondRetrieveDeduplicatesExistingSources:
    @pytest.mark.asyncio
    async def test_second_retrieve_dedups_existing_source(self, tmp_bibilab_home):
        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.retrieve", new_callable=AsyncMock) as mock_retrieve:
            # First call returns X, Y
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v1", video_title="Video One"),
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                ],
                [
                    FakeRetrievedChunk(
                        content="c1",
                        video_title="Video One",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v1",
                    )
                ],
            )
            registry: dict[str, CitationRegistryEntry] = {}
            source_map = {"v1": "s1", "v2": "s2"}

            await execute_retrieve(
                query="first",
                tool_name="retrieve",
                source_ids=["s1", "s2"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )

            # Second call returns Y, Z — Y already in registry
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                    FakeSourceHit(video_id="v3", video_title="Video Three"),
                ],
                [
                    FakeRetrievedChunk(
                        content="c2",
                        video_title="Video Two",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v2",
                    )
                ],
            )
            source_map3 = {"v2": "s2", "v3": "s3"}

            result = await execute_retrieve(
                query="second",
                tool_name="retrieve",
                source_ids=["s2", "s3"],
                cfg=cfg,
                registry=registry,
                source_map=source_map3,
            )

        # X kept index 1, Y kept index 2, Z got index 3
        assert registry["s1"].index == 1
        assert registry["s2"].index == 2
        assert registry["s3"].index == 3
        # Y's index unchanged after second call
        assert result["_turn_indices"] == [2, 3]


class TestChunkIdsAccumulateAcrossCalls:
    @pytest.mark.asyncio
    async def test_chunk_ids_accumulate_across_calls(self, tmp_bibilab_home):
        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.retrieve", new_callable=AsyncMock) as mock_retrieve:
            # First call: X at 0-30
            mock_retrieve.return_value = _make_result(
                [FakeSourceHit(video_id="v1", video_title="Video One")],
                [
                    FakeRetrievedChunk(
                        content="c1",
                        video_title="Video One",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v1",
                    )
                ],
            )
            registry: dict[str, CitationRegistryEntry] = {}
            source_map = {"v1": "s1"}

            await execute_retrieve(
                query="first",
                tool_name="retrieve",
                source_ids=["s1"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )

            assert registry["s1"].chunk_ids == {"v1_0_30"}

            # Second call: X at 60-90
            mock_retrieve.return_value = _make_result(
                [FakeSourceHit(video_id="v1", video_title="Video One")],
                [
                    FakeRetrievedChunk(
                        content="c2",
                        video_title="Video One",
                        timestamp_start=60.0,
                        timestamp_end=90.0,
                        video_id="v1",
                    )
                ],
            )

            await execute_retrieve(
                query="second",
                tool_name="retrieve",
                source_ids=["s1"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )

        assert registry["s1"].chunk_ids == {"v1_0_30", "v1_60_90"}


class TestTurnIndicesReflectsOnlyCurrentCall:
    @pytest.mark.asyncio
    async def test_turn_indices_reflects_only_current_call(self, tmp_bibilab_home):
        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.retrieve", new_callable=AsyncMock) as mock_retrieve:
            # First call: X, Y
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v1", video_title="Video One"),
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                ],
                [
                    FakeRetrievedChunk(
                        content="c1",
                        video_title="Video One",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v1",
                    ),
                    FakeRetrievedChunk(
                        content="c2",
                        video_title="Video Two",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v2",
                    ),
                ],
            )
            registry: dict[str, CitationRegistryEntry] = {}
            source_map = {"v1": "s1", "v2": "s2"}

            r1 = await execute_retrieve(
                query="first",
                tool_name="retrieve",
                source_ids=["s1", "s2"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )
            assert r1["_turn_indices"] == [1, 2]

            # Second call: only Y
            mock_retrieve.return_value = _make_result(
                [FakeSourceHit(video_id="v2", video_title="Video Two")],
                [
                    FakeRetrievedChunk(
                        content="c2",
                        video_title="Video Two",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v2",
                    )
                ],
            )

            r2 = await execute_retrieve(
                query="second",
                tool_name="retrieve",
                source_ids=["s2"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )
            # Turn indices only reflect sources retrieved in THIS call
            assert r2["_turn_indices"] == [2]


class TestSourceHeadersMatchRegistryStateAfterEachCall:
    @pytest.mark.asyncio
    async def test_source_headers_contain_all_registry_entries_after_each_call(self, tmp_bibilab_home):
        cfg = BibilabConfig(ai=AIConfig(protocol="openai", model="test", api_key="test", base_url=""))

        with patch("bibilab.pipeline.chat_tools.retrieve", new_callable=AsyncMock) as mock_retrieve:
            # First call: X, Y
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v1", video_title="Video One"),
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                ],
                [
                    FakeRetrievedChunk(
                        content="c1",
                        video_title="Video One",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v1",
                    )
                ],
            )
            registry: dict[str, CitationRegistryEntry] = {}
            source_map = {"v1": "s1", "v2": "s2"}

            r1 = await execute_retrieve(
                query="first",
                tool_name="retrieve",
                source_ids=["s1", "s2"],
                cfg=cfg,
                registry=registry,
                source_map=source_map,
            )

            headers1 = r1["_chunks"]
            assert 'Source [1]: "Video One"' in headers1
            assert 'Source [2]: "Video Two"' in headers1

            # Second call: Y, Z — headers must include X, Y, Z (full registry)
            mock_retrieve.return_value = _make_result(
                [
                    FakeSourceHit(video_id="v2", video_title="Video Two"),
                    FakeSourceHit(video_id="v3", video_title="Video Three"),
                ],
                [
                    FakeRetrievedChunk(
                        content="c2",
                        video_title="Video Two",
                        timestamp_start=0.0,
                        timestamp_end=30.0,
                        video_id="v2",
                    )
                ],
            )
            source_map3 = {"v2": "s2", "v3": "s3"}

            r2 = await execute_retrieve(
                query="second",
                tool_name="retrieve",
                source_ids=["s2", "s3"],
                cfg=cfg,
                registry=registry,
                source_map=source_map3,
            )

            headers2 = r2["_chunks"]
            # All three sources present — _build_source_headers walks full registry
            assert 'Source [1]: "Video One"' in headers2
            assert 'Source [2]: "Video Two"' in headers2
            assert 'Source [3]: "Video Three"' in headers2
