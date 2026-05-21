"""Tests for pipeline/embed.py"""

from __future__ import annotations

from bibilab.models._enums import _RELEVANCE_MARGIN_BY_MODE
from bibilab.pipeline.embed import (
    RetrievedChunk,
    _quantile_gate,
)


class TestQuantileGate:
    def test_quantile_gate_empty(self):
        """AC1-empty: empty pool returns empty."""
        assert _quantile_gate([], margin=2.0) == []

    def test_quantile_gate_all_negative_scores(self):
        """AC1-happy: chunks with all-negative scores keeps top (above 0 floor)."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                video_id="v1",
                distance=0.0,
                score=-5.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                video_id="v1",
                distance=0.0,
                score=-8.0,
            ),
        ]
        result = _quantile_gate(chunks, margin=2.0)
        # top=-5, median=-8, top-margin=-7 → threshold=max(0,-8,-7)=0
        # 0 floor is highest → keeps all chunks above 0
        # but no scores >= 0, so fallback to [chunks[0]]
        assert result == [chunks[0]]

    def test_quantile_gate_margin_various_values(self):
        """AC3: higher margin = more aggressive filtering (fewer chunks kept)."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                video_id="v1",
                distance=0.0,
                score=10.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                video_id="v1",
                distance=0.0,
                score=8.0,
            ),
            RetrievedChunk(
                content="c",
                video_title="t",
                timestamp_start=2,
                timestamp_end=3,
                video_id="v1",
                distance=0.0,
                score=6.0,
            ),
            RetrievedChunk(
                content="d",
                video_title="t",
                timestamp_start=3,
                timestamp_end=4,
                video_id="v1",
                distance=0.0,
                score=4.0,
            ),
        ]
        # margin=1.0: threshold = max(0, 6, 10-1=9) = 9 → only score >= 9 (10.0)
        r1 = _quantile_gate(chunks, margin=1.0)
        assert len(r1) == 1
        assert r1[0].score == 10.0

        # margin=2.0: threshold = max(0, 6, 10-2=8) = 8 → score >= 8 (10.0, 8.0)
        r2 = _quantile_gate(chunks, margin=2.0)
        assert len(r2) == 2
        assert [c.score for c in r2] == [10.0, 8.0]

        # margin=4.0: threshold = max(0, 6, 10-4=6) = 6 → median term dominates
        # score >= 6 (10.0, 8.0, 6.0); score=4.0 is below median so dropped
        r4 = _quantile_gate(chunks, margin=4.0)
        assert len(r4) == 3
        assert [c.score for c in r4] == [10.0, 8.0, 6.0]

    def test_quantile_gate_margin_3_many(self):
        """AC5-many: expected_hits='many' margin=3.0 filters less aggressively."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                video_id="v1",
                distance=0.0,
                score=10.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                video_id="v1",
                distance=0.0,
                score=7.5,
            ),
            RetrievedChunk(
                content="c",
                video_title="t",
                timestamp_start=2,
                timestamp_end=3,
                video_id="v1",
                distance=0.0,
                score=5.0,
            ),
            RetrievedChunk(
                content="d",
                video_title="t",
                timestamp_start=3,
                timestamp_end=4,
                video_id="v1",
                distance=0.0,
                score=2.5,
            ),
        ]
        margin = _RELEVANCE_MARGIN_BY_MODE["survey"]  # 2.5
        result = _quantile_gate(chunks, margin=margin)
        # top=10, median=5, threshold=max(0,5,10-3=7)=7 → keep score >= 7 (10, 7.5)
        assert len(result) == 2
        assert [c.score for c in result] == [10.0, 7.5]


class TestRelevanceMarginByModeMap:
    def test_relevance_margin_by_mode_map(self):
        """AC4: _RELEVANCE_MARGIN_BY_MODE maps mode to correct margins."""
        assert _RELEVANCE_MARGIN_BY_MODE["narrow"] == 2.0
        assert _RELEVANCE_MARGIN_BY_MODE["survey"] == 2.5


class TestRetrievalResultGateMargin:
    def test_retrieval_result_gate_margin(self):
        """AC7: RetrievalResult.gate_margin field exists and defaults to None."""
        from bibilab.pipeline.embed import RetrievalResult

        # Default: None when gate did not run
        r = RetrievalResult(
            chunks=[],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=1,
            source_coverage=[],
        )
        assert r.gate_margin is None

        # With gate_margin set
        r2 = RetrievalResult(
            chunks=[],
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=1,
            source_coverage=[],
            gate_margin=2.0,
        )
        assert r2.gate_margin == 2.0
