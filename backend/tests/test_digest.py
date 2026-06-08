"""Tests for the digest pipeline stage."""

import json

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    ContextWindowExceededError,
    LLMOutputBudgetExceededError,
)
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.digest import (
    _MAX_KEYWORDS,
    DigestResult,
    SectionDigest,
    _parse_response,
    _parse_section_digest_response,
    clean_str_facet,
    digest,
    parse_facet_int,
)


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


def test_digest_reinforcement_appears_in_prompt_en(mock_call_llm):
    """digest prepends and appends English language instruction so LLM cannot be biased by Chinese transcript."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    mock_call_llm.side_effect = capture_llm
    digest("中文transcript", _make_video_meta(), _make_ai_cfg("en"), output_language="en", ui_lang=None)

    assert captured is not None
    assert captured.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured


def test_digest_reinforcement_appears_in_prompt_zh(mock_call_llm):
    """digest with zh prepends and appends 简体中文 language instruction."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120):
        nonlocal captured
        captured = prompt
        return '{"summary": "一个视频。", "keywords": []}'

    mock_call_llm.side_effect = capture_llm
    digest("English transcript", _make_video_meta(), _make_ai_cfg("zh"), output_language="zh", ui_lang="zh")

    assert captured is not None
    assert captured.startswith("请用中文回答")
    assert "All output fields MUST be written in 简体中文" in captured


def test_digest_unknown_lang_falls_back_to_english(mock_call_llm):
    """digest with an unrecognized output_language falls back to English instruction and name."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    mock_call_llm.side_effect = capture_llm
    digest("transcript", _make_video_meta(), _make_ai_cfg("fr"), output_language="fr", ui_lang=None)

    assert captured is not None
    assert captured.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured


def test_digest_prompt_instructs_query_topic_keywords(mock_call_llm):
    """Keyword directive asks for title+content topical features and carries the cap."""
    captured = None

    def capture_llm(prompt, cfg, llm_timeout=120):
        nonlocal captured
        captured = prompt
        return '{"summary": "A video.", "keywords": []}'

    mock_call_llm.side_effect = capture_llm
    digest("transcript", _make_video_meta(), _make_ai_cfg())

    assert captured is not None
    assert f"up to {_MAX_KEYWORDS}, each 1-4 words" in captured
    assert "taken from title and transcript" in captured
    assert "a self-contained topic a viewer could ask a question about" in captured
    assert "Exclude three kinds of non-topics" in captured


class TestDigestResultShape:
    def test_digest_result_shape(self, mock_call_llm):
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

        mock_call_llm.return_value = mock_response
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

    def test_digest_keywords_capped_at_max(self, mock_call_llm):
        """Keywords returned by LLM are capped at _MAX_KEYWORDS."""
        overflow = [f"k{i}" for i in range(_MAX_KEYWORDS + 4)]
        mock_response = json.dumps({"summary": "A video.", "keywords": overflow})

        mock_call_llm.return_value = mock_response
        result = digest("transcript", _make_video_meta(), _make_ai_cfg())

        assert len(result.keywords) == _MAX_KEYWORDS
        assert result.keywords == overflow[:_MAX_KEYWORDS]

    def test_digest_llm_failure_raises_pipeline_error(self, mock_call_llm):
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

        mock_call_llm.side_effect = httpx.HTTPError("LLM down")
        with pytest.raises(PipelineError, match="LLM down"):
            digest("transcript", video_meta, ai_cfg)

    def test_digest_retry_on_budget_error_then_succeed(self, mock_call_llm):
        """Output budget exhaustion is a structural failure (identical across
        attempts with the same prompt + budget), so the digest loop should
        re-raise immediately rather than waste 2 more calls. Verifies the new
        LLMOutputBudgetExceededError is in the re-raise list."""

        video_meta = VideoMeta(
            video_id="budget",
            title="T",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/budget",
            cover_url="",
            duration_seconds=10,
            uploader="U",
        )
        ai_cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )
        mock_call_llm.side_effect = LLMOutputBudgetExceededError("exhausted")
        with pytest.raises(LLMOutputBudgetExceededError, match="exhausted"):
            digest("transcript", video_meta, ai_cfg)
        # Only 1 call — the loop bails out on the first budget error.
        assert mock_call_llm.call_count == 1

    def test_digest_retry_on_context_overflow(self, mock_call_llm):
        """Input overflow is non-retryable (input won't shrink between calls).
        The digest loop should re-raise immediately on the first
        ContextWindowExceededError."""
        video_meta = VideoMeta(
            video_id="overflow",
            title="T",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/overflow",
            cover_url="",
            duration_seconds=10,
            uploader="U",
        )
        ai_cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )
        mock_call_llm.side_effect = ContextWindowExceededError("overflow")
        with pytest.raises(ContextWindowExceededError, match="overflow"):
            digest("transcript", video_meta, ai_cfg)
        # Only 1 call — no retry.
        assert mock_call_llm.call_count == 1


class TestDigestResultFacets:
    def test_facets_default_to_none(self):
        result = DigestResult(summary="S", keywords=["k"])
        assert result.series_name is None
        assert result.sequence_number is None
        assert result.season_number is None

    def test_facets_populated_from_json(self):
        json_str = (
            '{"summary": "S", "keywords": ["k"], '
            '"series_name": "罗翔说刑法", "sequence_number": 8, '
            '"season_number": null}'
        )
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.keywords == ["k"]
        assert result.series_name == "罗翔说刑法"
        assert result.sequence_number == 8
        assert result.season_number is None

    def test_facets_missing_keys_default_to_none(self):
        json_str = '{"summary": "S", "keywords": []}'
        result = _parse_response(json_str)
        assert result.series_name is None
        assert result.sequence_number is None
        assert result.season_number is None

    def test_sequence_number_coerces_string_to_int(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": "8"}'
        result = _parse_response(json_str)
        assert result.sequence_number == 8
        assert isinstance(result.sequence_number, int)

    def test_sequence_number_coerces_float_to_int(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8.0}'
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
        json_str = '{"summary": "S", "keywords": [], "sequence_number": "第八集"}'
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.sequence_number is None

    def test_sequence_number_fractional_float_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8.5}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_sequence_number_below_one_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 0}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_sequence_number_boolean_degrades_to_none(self):
        json_str = '{"summary": "S", "keywords": [], "sequence_number": true}'
        result = _parse_response(json_str)
        assert result.sequence_number is None

    def test_bare_sequence_number_persists_after_removal(self):
        # AC3: after removing _require_kind_with_number, a bare sequence_number
        # with no sequence_kind persists (was previously dropped).
        json_str = '{"summary": "S", "keywords": [], "sequence_number": 8}'
        result = _parse_response(json_str)
        assert result.sequence_number == 8

    def test_sequence_kind_not_in_model(self):
        # DigestResult no longer has sequence_kind after its removal.
        result = DigestResult(summary="S", keywords=["k"])
        assert not hasattr(result, "sequence_kind")

    def test_full_digest_pipeline_includes_facets(self, mock_call_llm):
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
            '"season_number": null}'
        )

        mock_call_llm.return_value = mock_response
        result = digest("transcript about law", video_meta, ai_cfg)

        assert result.series_name == "罗翔说刑法"
        assert result.sequence_number == 8
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
        json_str = '{"summary": "S", "keywords": [], "sequence_number": Infinity}'
        result = _parse_response(json_str)
        assert result.summary == "S"
        assert result.sequence_number is None


class TestDigestRetry:
    def test_bad_facet_does_not_trigger_retry(self, mock_call_llm):
        # A malformed facet degrades to None in-place; the digest is good and
        # must be returned on the first call — no retry, no PipelineError.
        bad_facet = '{"summary": "S", "keywords": ["k"], "sequence_number": "第八集"}'
        mock_call_llm.return_value = bad_facet
        result = digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert mock_call_llm.call_count == 1
        assert result.summary == "S"
        assert result.sequence_number is None

    def test_transient_http_error_recovers_on_retry(self, mock_call_llm):
        import httpx

        valid = '{"summary": "recovered", "keywords": ["k"]}'
        mock_call_llm.side_effect = [httpx.HTTPError("transient"), valid]
        result = digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert mock_call_llm.call_count == 2
        assert result.summary == "recovered"

    def test_missing_summary_exhausts_retries_raises_pipeline_error(self, mock_call_llm):
        # A genuine schema failure (required summary absent) must retry the
        # bounded number of times and then raise — never silently succeed.
        no_summary = '{"keywords": ["k"]}'
        mock_call_llm.return_value = no_summary
        with pytest.raises(PipelineError, match="exhausted all retries"):
            digest("transcript", _make_video_meta(), _make_ai_cfg())
        assert mock_call_llm.call_count == 3


class TestParseFacetInt:
    def test_none_and_blank_return_none(self):
        assert parse_facet_int(None) is None
        assert parse_facet_int("") is None
        assert parse_facet_int("   ") is None

    def test_valid_int_float_str(self):
        assert parse_facet_int(8) == 8
        assert parse_facet_int(8.0) == 8
        assert parse_facet_int(" 12 ") == 12

    def test_invalid_raises_valueerror(self):
        for bad in ["第八集", "abc", 0, -1, 8.5, True, False, ["x"], {}]:
            with pytest.raises(ValueError):
                parse_facet_int(bad)

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            parse_facet_int(float("inf"))


class TestCleanStrFacet:
    def test_none_returns_none(self):
        assert clean_str_facet(None) is None

    def test_trim_and_blank(self):
        assert clean_str_facet("  罗翔说刑法  ") == "罗翔说刑法"
        assert clean_str_facet("") is None
        assert clean_str_facet("   ") is None

    def test_non_string_raises(self):
        for bad in [5, True, 1.5, ["x"], {}]:
            with pytest.raises(ValueError):
                clean_str_facet(bad)


class TestSectionDigest:
    def test_section_digest_parses_summary_and_keywords(self):
        json_str = '{"summary": "A section.", "keywords": ["x", "y"]}'
        sd = _parse_section_digest_response(json_str)
        assert isinstance(sd, SectionDigest)
        assert sd.summary == "A section."
        assert sd.keywords == ["x", "y"]

    def test_section_digest_has_no_facet_fields(self):
        # SectionDigest is the per-section view; facets are source-level
        # (extracted once at section 1 on DigestResult). They MUST NOT appear
        # on SectionDigest.
        assert not hasattr(SectionDigest, "series_name")
        assert not hasattr(SectionDigest, "sequence_number")
        assert not hasattr(SectionDigest, "season_number")

    def test_section_digest_keywords_capped_at_max(self):
        overflow = [f"k{i}" for i in range(_MAX_KEYWORDS + 4)]
        json_str = json.dumps({"summary": "S", "keywords": overflow})
        sd = _parse_section_digest_response(json_str)
        assert len(sd.keywords) == _MAX_KEYWORDS
        assert sd.keywords == overflow[:_MAX_KEYWORDS]
