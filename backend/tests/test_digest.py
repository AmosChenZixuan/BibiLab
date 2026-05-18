"""Tests for the digest pipeline stage."""

import json
from unittest.mock import patch

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.digest import _MAX_KEYWORDS, DigestResult, _parse_response, digest


def _make_video_meta(title="Test Video") -> VideoMeta:
    return VideoMeta(
        video_id="test123",
        title=title,
        platform="bilibili",
        source_url="https://bilibili.example.com/video/test123",
        cover_url="",
        duration_seconds=600,
        uploader="TestUploader",
    )


def _make_ai_cfg(output_language="en") -> AIConfig:
    return AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language=output_language,
    )


def test_digest_reinforcement_appears_in_prompt_en():
    """digest prepends and appends English language instruction so LLM cannot be biased by Chinese transcript."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    with patch("bibilab.pipeline.digest._call_llm", side_effect=capture_llm):
        digest("中文transcript", _make_video_meta(), _make_ai_cfg("en"), output_language="en", ui_lang=None)

    assert captured is not None
    assert captured.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured


def test_digest_reinforcement_appears_in_prompt_zh():
    """digest with zh prepends and appends Chinese language instruction."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured
        captured = prompt
        return '{"summary": "一个视频。", "keywords": []}'

    with patch("bibilab.pipeline.digest._call_llm", side_effect=capture_llm):
        digest("English transcript", _make_video_meta(), _make_ai_cfg("zh"), output_language="zh", ui_lang="zh")

    assert captured is not None
    assert captured.startswith("请用中文回答")
    assert "All output fields MUST be written in Chinese" in captured


def test_digest_unknown_lang_falls_back_to_english():
    """digest with an unrecognized output_language falls back to English instruction and name."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    with patch("bibilab.pipeline.digest._call_llm", side_effect=capture_llm):
        digest("transcript", _make_video_meta(), _make_ai_cfg("fr"), output_language="fr", ui_lang=None)

    assert captured is not None
    assert captured.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured


def test_digest_prompt_instructs_query_topic_keywords():
    """Keyword directive asks for title+content topical features and carries the cap."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    with patch("bibilab.pipeline.digest._call_llm", side_effect=capture_llm):
        digest("transcript", _make_video_meta(), _make_ai_cfg())

    assert captured is not None
    assert f"up to {_MAX_KEYWORDS}, each 1-4 words" in captured
    assert "taken from title and transcript" in captured
    assert "a self-contained topic a viewer could ask a question about" in captured
    assert "Exclude three kinds of non-topics" in captured


class TestDigestResultShape:
    def test_digest_result_shape(self):
        """DigestResult has summary + keywords, no title/key_points, keywords capped at 5."""
        video_meta = VideoMeta(
            video_id="test123",
            title="Test Video",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/test123",
            cover_url="",
            duration_seconds=600,
            uploader="TestUploader",
        )
        ai_cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )
        transcript = "This is a test transcript."

        mock_response = '{"summary": "A great video about ML.", "keywords": ["machine learning", "neural nets"]}'

        with patch("bibilab.pipeline.digest._call_llm", return_value=mock_response):
            result = digest(transcript, video_meta, ai_cfg)

        assert isinstance(result.summary, str)
        assert result.summary == "A great video about ML."
        assert isinstance(result.keywords, list)
        assert all(isinstance(k, str) for k in result.keywords)
        assert len(result.keywords) <= 5
        assert result.keywords == ["machine learning", "neural nets"]

        # No title or key_points attributes
        assert not hasattr(result, "title")
        assert not hasattr(result, "key_points")

    def test_digest_keywords_capped_at_max(self):
        """Keywords returned by LLM are capped at _MAX_KEYWORDS."""
        overflow = [f"k{i}" for i in range(_MAX_KEYWORDS + 4)]
        mock_response = json.dumps({"summary": "A video.", "keywords": overflow})

        with patch("bibilab.pipeline.digest._call_llm", return_value=mock_response):
            result = digest("transcript", _make_video_meta(), _make_ai_cfg())

        assert len(result.keywords) == _MAX_KEYWORDS
        assert result.keywords == overflow[:_MAX_KEYWORDS]

    def test_digest_llm_failure_raises_pipeline_error(self):
        """On LLM failure (all retries exhausted), raises PipelineError instead of silent data loss."""
        video_meta = VideoMeta(
            video_id="test789",
            title="Test Video",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/test789",
            cover_url="",
            duration_seconds=300,
            uploader="TestUploader",
        )
        ai_cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )
        # Use httpx.HTTPError which is a retriable exception
        import httpx

        with patch("bibilab.pipeline.digest._call_llm", side_effect=httpx.HTTPError("LLM down")):
            with pytest.raises(PipelineError, match="LLM down"):
                digest("transcript", video_meta, ai_cfg)


class TestDigestResultFacets:
    def test_facets_default_to_none(self):
        result = DigestResult(summary="S", keywords=["k"])
        assert result.series_name is None
        assert result.sequence_number is None
        assert result.sequence_kind is None
        assert result.season_number is None

    def test_facets_populated_from_json(self):
        json_str = (
            '{"summary": "S", "keywords": ["k"], '
            '"series_name": "罗翔说刑法", "sequence_number": 8, '
            '"sequence_kind": "episode", "season_number": null}'
        )
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.keywords == ["k"]
        assert result.series_name == "罗翔说刑法"
        assert result.sequence_number == 8
        assert result.sequence_kind == "episode"
        assert result.season_number is None

    def test_facets_missing_keys_default_to_none(self):
        json_str = '{"summary": "S", "keywords": []}'
        result = _parse_response(json_str)
        assert result.series_name is None
        assert result.sequence_number is None
        assert result.sequence_kind is None
        assert result.season_number is None

    def test_sequence_number_coerces_string_to_int(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": "8", "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number == 8
        assert isinstance(result.sequence_number, int)

    def test_sequence_number_coerces_float_to_int(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8.0, "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number == 8
        assert isinstance(result.sequence_number, int)

    def test_season_number_coerces_string_to_int(self):
        json_str = '{"summary": "S", "keywords": [], "season_number": "2"}'
        result = _parse_response(json_str)
        assert result.season_number == 2
        assert isinstance(result.season_number, int)

    def test_sequence_number_non_numeric_degrades_to_none(self):
        # Unparseable facet must not abort the digest — degrade to None, keep summary.
        json_str = '{"summary": "S", "keywords": [], "sequence_number": "第八集", "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.sequence_number is None
        assert result.sequence_kind is None  # co-occurrence drops the orphaned kind

    def test_sequence_number_fractional_float_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8.5, "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_sequence_number_below_one_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 0, "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_sequence_number_boolean_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": true, "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_sequence_kind_lowercased(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 1, "sequence_kind": "Episode"}'
        result = _parse_response(json_str)
        assert result.sequence_kind == "episode"

    def test_sequence_kind_free_form_accepted(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 1, "sequence_kind": "volume"}'
        result = _parse_response(json_str)
        assert result.sequence_kind == "volume"

    def test_sequence_kind_non_string_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 1, "sequence_kind": 5}'
        result = _parse_response(json_str)
        assert result.sequence_kind is None
        assert result.sequence_number is None  # co-occurrence drops the orphaned number

    def test_sequence_kind_empty_string_becomes_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_kind": ""}'
        result = _parse_response(json_str)
        assert result.sequence_kind is None

    def test_sequence_number_without_kind_drops_both(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8}'
        result = _parse_response(json_str)
        assert result.sequence_number is None
        assert result.sequence_kind is None

    def test_sequence_kind_without_number_drops_both(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.sequence_number is None
        assert result.sequence_kind is None

    def test_full_digest_pipeline_includes_facets(self):
        video_meta = VideoMeta(
            video_id="test123",
            title="罗翔说刑法 EP08 正当防卫",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/test123",
            cover_url="",
            duration_seconds=600,
            uploader="罗翔说刑法",
        )
        ai_cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )

        mock_response = (
            '{"summary": "A lecture on law.", "keywords": ["law", "criminal"], '
            '"series_name": "罗翔说刑法", "sequence_number": 8, '
            '"sequence_kind": "episode", "season_number": null}'
        )

        with patch("bibilab.pipeline.digest._call_llm", return_value=mock_response):
            result = digest("transcript about law", video_meta, ai_cfg)

        assert result.series_name == "罗翔说刑法"
        assert result.sequence_number == 8
        assert result.sequence_kind == "episode"
        assert result.season_number is None

    def test_series_name_non_string_degrades_to_none(self):
        # A non-string series_name must degrade to None, not abort the digest.
        json_str = '{"summary": "S", "keywords": [], "series_name": 123}'
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.series_name is None

    def test_series_name_boolean_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "series_name": false}'
        result = _parse_response(json_str)
        assert result.series_name is None

    def test_series_name_empty_string_becomes_none(self):
        json_str = '{"summary": "S", "keywords": [], "series_name": "   "}'
        result = _parse_response(json_str)
        assert result.series_name is None

    def test_series_name_whitespace_trimmed(self):
        json_str = '{"summary": "S", "keywords": [], "series_name": "  罗翔说刑法  "}'
        result = _parse_response(json_str)
        assert result.series_name == "罗翔说刑法"

    def test_sequence_number_non_finite_degrades_to_none(self):
        # json.loads accepts bare Infinity/NaN — must degrade, not raise.
        json_str = '{"summary": "S", "keywords": [], "sequence_number": Infinity, "sequence_kind": "episode"}'
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.sequence_number is None
        assert result.sequence_kind is None  # co-occurrence drops the orphan


class TestDigestRetry:
    def test_bad_facet_does_not_trigger_retry(self):
        # A malformed facet degrades to None in-place; the digest is good and
        # must be returned on the first call — no retry, no PipelineError.
        bad_facet = '{"summary": "S", "keywords": ["k"], "sequence_number": "第八集", "sequence_kind": "episode"}'
        with patch("bibilab.pipeline.digest._call_llm", return_value=bad_facet) as m:
            result = digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert m.call_count == 1
        assert result.summary == "S"
        assert result.sequence_number is None
        assert result.sequence_kind is None

    def test_transient_http_error_recovers_on_retry(self):
        import httpx

        valid = '{"summary": "recovered", "keywords": ["k"]}'
        with patch(
            "bibilab.pipeline.digest._call_llm",
            side_effect=[httpx.HTTPError("transient"), valid],
        ) as m:
            result = digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert m.call_count == 2
        assert result.summary == "recovered"

    def test_missing_summary_exhausts_retries_raises_pipeline_error(self):
        # A genuine schema failure (required summary absent) must retry the
        # bounded number of times and then raise — never silently succeed.
        no_summary = '{"keywords": ["k"]}'
        with patch("bibilab.pipeline.digest._call_llm", return_value=no_summary) as m:
            with pytest.raises(PipelineError, match="exhausted all retries"):
                digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert m.call_count == 3
