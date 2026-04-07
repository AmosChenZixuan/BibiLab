"""Tests for the digest pipeline stage."""

from unittest.mock import patch

import pytest

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.digest import digest


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
            provider="openai",
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

    def test_digest_keywords_capped_at_5(self):
        """Keywords returned by LLM are capped at 5."""
        video_meta = VideoMeta(
            video_id="test456",
            title="Test Video",
            platform="bilibili",
            source_url="https://bilibili.example.com/video/test456",
            cover_url="",
            duration_seconds=300,
            uploader="TestUploader",
        )
        ai_cfg = AIConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )
        # Return more than 5 keywords
        mock_response = '{"summary": "A video.", "keywords": ["a", "b", "c", "d", "e", "f", "g"]}'

        with patch("bibilab.pipeline.digest._call_llm", return_value=mock_response):
            result = digest("transcript", video_meta, ai_cfg)

        assert len(result.keywords) == 5
        assert result.keywords == ["a", "b", "c", "d", "e"]

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
            provider="openai",
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
