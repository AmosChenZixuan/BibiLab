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
        """All-negative scores → relative threshold keeps highest score, drops rest."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                source_id="v1",
                distance=0.0,
                score=-5.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                source_id="v1",
                distance=0.0,
                score=-8.0,
            ),
        ]
        result = _quantile_gate(chunks, margin=2.0)
        # top=-5, median=-8, top-margin=-7 → threshold=max(-8,-7)=-5
        # score -5.0 >= -5 → keep 'a'; -8.0 < -5 → drop 'b'
        assert len(result) == 1
        assert result[0].score == -5.0

    def test_quantile_gate_margin_various_values(self):
        """AC3: higher margin = more aggressive filtering (fewer chunks kept)."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                source_id="v1",
                distance=0.0,
                score=10.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                source_id="v1",
                distance=0.0,
                score=8.0,
            ),
            RetrievedChunk(
                content="c",
                video_title="t",
                timestamp_start=2,
                timestamp_end=3,
                source_id="v1",
                distance=0.0,
                score=6.0,
            ),
            RetrievedChunk(
                content="d",
                video_title="t",
                timestamp_start=3,
                timestamp_end=4,
                source_id="v1",
                distance=0.0,
                score=4.0,
            ),
        ]
        # margin=1.0: threshold = max(6, 10-1=9) = 9 → only score >= 9 (10.0)
        r1 = _quantile_gate(chunks, margin=1.0)
        assert len(r1) == 1
        assert r1[0].score == 10.0

        # margin=2.0: threshold = max(6, 10-2=8) = 8 → score >= 8 (10.0, 8.0)
        r2 = _quantile_gate(chunks, margin=2.0)
        assert len(r2) == 2
        assert [c.score for c in r2] == [10.0, 8.0]

        # margin=4.0: threshold = max(6, 10-4=6) = 6 → median term dominates
        # score >= 6 (10.0, 8.0, 6.0); score=4.0 is below median so dropped
        r4 = _quantile_gate(chunks, margin=4.0)
        assert len(r4) == 3
        assert [c.score for c in r4] == [10.0, 8.0, 6.0]

    def test_quantile_gate_margin_survey(self):
        """AC5-survey: mode='survey' margin=2.5 filters less aggressively."""
        chunks = [
            RetrievedChunk(
                content="a",
                video_title="t",
                timestamp_start=0,
                timestamp_end=1,
                source_id="v1",
                distance=0.0,
                score=10.0,
            ),
            RetrievedChunk(
                content="b",
                video_title="t",
                timestamp_start=1,
                timestamp_end=2,
                source_id="v1",
                distance=0.0,
                score=7.5,
            ),
            RetrievedChunk(
                content="c",
                video_title="t",
                timestamp_start=2,
                timestamp_end=3,
                source_id="v1",
                distance=0.0,
                score=5.0,
            ),
            RetrievedChunk(
                content="d",
                video_title="t",
                timestamp_start=3,
                timestamp_end=4,
                source_id="v1",
                distance=0.0,
                score=2.5,
            ),
        ]
        margin = _RELEVANCE_MARGIN_BY_MODE["survey"]  # 2.5
        result = _quantile_gate(chunks, margin=margin)
        # top=10, median=5, threshold=max(5, 10-2.5=7.5)=7.5 → keep score >= 7.5 (10, 7.5)
        assert len(result) == 2
        assert [c.score for c in result] == [10.0, 7.5]


class TestRelevanceMarginByModeMap:
    def test_relevance_margin_by_mode_map(self):
        """AC4: _RELEVANCE_MARGIN_BY_MODE maps mode to correct margins."""
        assert _RELEVANCE_MARGIN_BY_MODE["narrow"] == 2.0
        assert _RELEVANCE_MARGIN_BY_MODE["survey"] == 2.5


class TestToolNameToParamsMapping:
    def test_tool_name_to_params(self):
        """AC2: _TOOL_NAME_TO_PARAMS maps tool names to RetrievalParams."""
        from bibilab.models._enums import RetrievalParams
        from bibilab.pipeline.chat_tools import _TOOL_NAME_TO_PARAMS

        r = _TOOL_NAME_TO_PARAMS["retrieve"]
        assert r == RetrievalParams(depth_per_source=2, top_k=8, mode="narrow")

        s = _TOOL_NAME_TO_PARAMS["survey"]
        assert s == RetrievalParams(depth_per_source=5, top_k=24, mode="survey")

        sc = _TOOL_NAME_TO_PARAMS["retrieve_scoped"]
        assert sc == RetrievalParams(depth_per_source=2, top_k=8, mode="narrow")


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


class _MockEncoding:
    """Minimal tokenizer encoding that ONNXMultilingualEmbedding.__call__ expects."""

    def __init__(self, ids: list[int], attention_mask: list[int], type_ids: list[int]):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


def _make_mock_tokenizer():
    from unittest.mock import MagicMock

    def fake_encode(text):
        n = max(len(text), 1)
        return _MockEncoding(list(range(n)), [1] * n, [0] * n)

    tokenizer = MagicMock()
    tokenizer.encode = fake_encode
    return tokenizer


def _make_mock_session(dim=384):
    from unittest.mock import MagicMock

    import numpy as np

    def fake_run(_, onnx_input):
        batch_size = onnx_input["input_ids"].shape[0]
        seq_len = onnx_input["input_ids"].shape[1]
        return [np.random.randn(batch_size, seq_len, dim).astype(np.float32)]

    session = MagicMock()
    session.run = fake_run
    return session


class TestONNXMultilingualEmbedding:
    def test_multilingual_embedding_dimension(self):
        """Multilingual model returns 384-dim vectors for mixed English/Chinese input."""
        import math
        from unittest.mock import patch

        from bibilab.pipeline.embed import ONNXMultilingualEmbedding

        mock_session = _make_mock_session()
        mock_tokenizer = _make_mock_tokenizer()

        with (
            patch("bibilab.pipeline.embed.ensure"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
            patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer),
        ):
            emb = ONNXMultilingualEmbedding()
            result = emb(["hello world", "你好世界"])
        assert len(result) == 2
        assert len(result[0]) == 384
        assert all(math.isfinite(v) for v in result[0])
        assert all(math.isfinite(v) for v in result[1])
        assert result[0] != result[1]

    def test_multilingual_embedding_empty_input(self):
        """Empty input returns empty list without touching ONNX session."""
        from unittest.mock import patch

        from bibilab.pipeline.embed import ONNXMultilingualEmbedding

        with (
            patch("bibilab.pipeline.embed.ensure"),
            patch("onnxruntime.InferenceSession"),
            patch("tokenizers.Tokenizer.from_file"),
        ):
            emb = ONNXMultilingualEmbedding()
            assert emb([]) == []

    def test_multilingual_embedding_download_check(self):
        """is_embedding_model_downloaded returns a bool."""
        from bibilab.pipeline.embed import is_embedding_model_downloaded

        assert isinstance(is_embedding_model_downloaded(), bool)


class TestRowFromChromaSegRange:
    def test_row_from_chroma_hydrates_seg_range(self):
        from bibilab.pipeline.embed import _row_from_chroma

        chunk = _row_from_chroma(
            content="x",
            metadata={"source_id": "s1", "seg_start": 4, "seg_end": 9, "video_title": "t"},
            distance=0.1,
            chroma_id="s1_2",
        )
        assert chunk.seg_start == 4
        assert chunk.seg_end == 9

    def test_row_from_chroma_seg_range_defaults_when_absent(self):
        from bibilab.pipeline.embed import _row_from_chroma

        chunk = _row_from_chroma(content="x", metadata={"source_id": "s1"}, distance=0.1, chroma_id="s1_0")
        assert chunk.seg_start is None
        assert chunk.seg_end is None
