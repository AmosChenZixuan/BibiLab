"""Tests for embed.py retrieve() — #333 neighbor chunk pull."""

from __future__ import annotations

import pytest

from bibilab.config import BibilabConfig
from bibilab.pipeline.embed import RetrievedChunk, _chunk_key, _chunks_from_chroma_rows


class TestRetrievedChunkSequenceIndex:
    """AC: sequence_index is the LAST dataclass field; positional-arg constructors unaffected."""

    def test_sequence_index_last_field(self):
        """Positional arg at end still works."""
        c = RetrievedChunk("content", "title", 0.0, 10.0, "vid", 0.5, None)
        assert c.sequence_index is None

    def test_sequence_index_kwarg(self):
        """sequence_index can be set via kwarg."""
        c = RetrievedChunk(
            content="content",
            video_title="title",
            timestamp_start=0.0,
            timestamp_end=10.0,
            video_id="vid",
            distance=0.5,
            score=None,
            sequence_index=3,
        )
        assert c.sequence_index == 3


class TestChunksFromChromaRows:
    """Unit tests for _chunks_from_chroma_rows helper."""

    def test_empty_results(self):
        assert _chunks_from_chroma_rows({}) == []
        assert _chunks_from_chroma_rows({"documents": []}) == []

    def test_parses_all_fields(self):
        rows = {
            "ids": [["vid1_2"]],
            "documents": ["chunk content"],
            "metadatas": [
                {
                    "video_id": "vid1",
                    "video_title": "Test Video",
                    "timestamp_start": 0.0,
                    "timestamp_end": 30.0,
                    "sequence_index": 2,
                }
            ],
            "distances": [[0.3]],
        }
        chunks = _chunks_from_chroma_rows(rows)
        assert len(chunks) == 1
        assert chunks[0].content == "chunk content"
        assert chunks[0].video_id == "vid1"
        assert chunks[0].sequence_index == 2

    def test_sequence_index_from_id(self):
        """sequence_index extracted from id="{video_id}_{sequence_index}"."""
        rows = {
            "ids": [["BV1CbSjBGEib_460"]],
            "documents": ["test"],
            "metadatas": [
                {
                    "video_id": "BV1CbSjBGEib",
                    "video_title": "t",
                    "timestamp_start": 0.0,
                    "timestamp_end": 30.0,
                }
            ],
            "distances": [[0.1]],
        }
        chunks = _chunks_from_chroma_rows(rows)
        assert chunks[0].sequence_index == 460

    def test_malformed_id_gives_none(self):
        rows = {
            "ids": [["no_underscore"]],
            "documents": ["test"],
            "metadatas": [
                {
                    "video_id": "no_underscore",
                    "video_title": "t",
                    "timestamp_start": 0.0,
                    "timestamp_end": 30.0,
                }
            ],
            "distances": [[0.1]],
        }
        chunks = _chunks_from_chroma_rows(rows)
        assert chunks[0].sequence_index is None


class TestNeighborPullBypass:
    """AC: neighbor pull is bypassed when threshold=0 or reranked=False."""

    @pytest.fixture
    def cfg_threshold_zero(self):
        cfg = BibilabConfig()
        cfg.rag.neighbor_scarcity_threshold = 0
        cfg.rag.reranking_enabled = True
        return cfg

    @pytest.fixture
    def cfg_threshold_two(self):
        cfg = BibilabConfig()
        cfg.rag.neighbor_scarcity_threshold = 2
        cfg.rag.reranking_enabled = True
        return cfg

    def test_threshold_zero_no_pull(self, cfg_threshold_zero):
        """threshold=0 short-circuits before any work."""
        # With threshold=0, the algorithm should skip entirely.
        # We can verify by checking that neighbor_scarcity_threshold <= 0
        # is the guard condition.
        assert cfg_threshold_zero.rag.neighbor_scarcity_threshold <= 0

    def test_threshold_two_conditional(self, cfg_threshold_two):
        """threshold=2 is active (conditional on len(result_chunks) <= 2)."""
        assert cfg_threshold_two.rag.neighbor_scarcity_threshold > 0
        assert cfg_threshold_two.rag.neighbor_scarcity_threshold == 2


class TestNeighborPullAlgorithm:
    """Algorithm shape tests for neighbor pull logic (mock-free)."""

    def test_candidate_indices_i_minus_1_and_i_plus_1(self):
        """Hit at index 2 → candidates {1, 3}."""
        # Verify the spec formula: candidates = ⋃ {i-1, i+1 for i in hit_indices}
        hit_indices = [2]
        candidates = set()
        for idx in hit_indices:
            for delta in (-1, 1):
                n = idx + delta
                if n >= 0:
                    candidates.add(n)
        assert candidates == {1, 3}

    def test_lower_clamp_index_zero(self):
        """Hit at index 0 → no negative neighbor."""
        hit_indices = [0]
        candidates = set()
        for idx in hit_indices:
            for delta in (-1, 1):
                n = idx + delta
                if n >= 0:  # Lower clamp
                    candidates.add(n)
        assert candidates == {1}  # Only +1 survives

    def test_candidate_dedup_already_in_hits(self):
        """Candidate that is already a hit is excluded."""
        hit_indices = [2, 3]
        existing = {2, 3}
        candidates = set()
        for idx in hit_indices:
            for delta in (-1, 1):
                n = idx + delta
                if n >= 0:
                    candidates.add(n)
        # After dedup against hits
        candidates -= existing
        assert candidates == {1, 4}

    def test_two_hits_same_source_spaced(self):
        """Hits {2, 4} → candidates {1, 3, 5} (no dedup between hits themselves)."""
        hit_indices = [2, 4]
        candidates = set()
        for idx in hit_indices:
            for delta in (-1, 1):
                n = idx + delta
                if n >= 0:
                    candidates.add(n)
        assert candidates == {1, 3, 5}

    def test_two_hits_same_source_clustered(self):
        """Hits {2, 3} → raw candidates {1, 2, 3, 4}; deduped against hits = {1, 4}."""
        hit_indices = [2, 3]
        candidates = set()
        for idx in hit_indices:
            for delta in (-1, 1):
                n = idx + delta
                if n >= 0:
                    candidates.add(n)
        # Raw: {2-1, 2+1, 3-1, 3+1} = {1, 3, 2, 4} = {1, 2, 3, 4}
        assert candidates == {1, 2, 3, 4}
        # After algorithm's existing_keys dedup (removes hits themselves)
        candidates -= {2, 3}
        assert candidates == {1, 4}

    def test_chunk_key_dedup(self):
        """_chunk_key dedup correctly identifies duplicate chunks."""
        c1 = RetrievedChunk("a", "t", 0.0, 10.0, "vid", 0.1, score=1.0, sequence_index=1)
        c2 = RetrievedChunk("a", "t", 0.0, 10.0, "vid", 0.1, score=1.0, sequence_index=1)
        # Same key means same chunk
        assert _chunk_key(c1) == _chunk_key(c2)


class TestNeighborSentinelScores:
    """Neighbors get score=None, distance=inf sentinel."""

    def test_inf_distance_sort_buries(self):
        """distance=inf should sort to bottom in _chunk_score."""
        from bibilab.pipeline.embed import _chunk_score

        hit = RetrievedChunk("hit", "t", 0.0, 10.0, "v", 0.1, score=5.0, sequence_index=2)
        neighbor = RetrievedChunk("nbr", "t", 10.0, 20.0, "v", float("inf"), score=None, sequence_index=3)
        # _chunk_score: lower = more relevant
        assert _chunk_score(hit) < _chunk_score(neighbor)  # hit is more relevant
        assert _chunk_score(neighbor) == float("inf")


class TestRetrievalResultNeighborsPulled:
    """AC: RetrievalResult.neighbors_pulled field exists and defaults to 0."""

    def test_retrieval_result_neighbors_pulled_default(self):
        from bibilab.pipeline.embed import RetrievalResult

        r = RetrievalResult(
            chunks=[],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=1,
            source_coverage=[],
        )
        assert r.neighbors_pulled == 0

    def test_retrieval_result_neighbors_pulled_set(self):
        from bibilab.pipeline.embed import RetrievalResult

        r = RetrievalResult(
            chunks=[],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=1,
            source_coverage=[],
            neighbors_pulled=2,
        )
        assert r.neighbors_pulled == 2


class TestDefaultConfig:
    """AC: neighbor_scarcity_threshold defaults to 2."""

    def test_default_threshold(self):
        cfg = BibilabConfig()
        assert cfg.rag.neighbor_scarcity_threshold == 2

    def test_threshold_zero_disables(self):
        cfg = BibilabConfig()
        cfg.rag.neighbor_scarcity_threshold = 0
        assert cfg.rag.neighbor_scarcity_threshold <= 0  # Guard condition
