"""Tests for LLM client caching."""

from unittest.mock import MagicMock, patch

import anthropic
import openai

from bibilab.config import AIConfig


class TestAnthropicClientCaching:
    """Verify that the Anthropic client is reused across calls, not recreated each time."""

    def test_anthropic_client_reused_across_calls(self):
        """After first call, subsequent calls reuse the same client instance."""
        from bibilab.pipeline._shared import _call_llm

        cfg = AIConfig(
            protocol="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            base_url="",
            output_language="en",
        )

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(type="text", text="test response")]
        )

        with patch.object(anthropic, "Anthropic", return_value=mock_client_instance) as mock_anthropic_cls:
            # First call
            result1 = _call_llm("prompt 1", cfg)
            # Second call
            result2 = _call_llm("prompt 2", cfg)

        # Client should be created only once and reused
        assert mock_anthropic_cls.call_count == 1, f"Expected 1 client creation, got {mock_anthropic_cls.call_count}"
        # Both calls should use the same client instance
        assert result1 == "test response"
        assert result2 == "test response"


class TestOpenAIClientCaching:
    """Verify that the OpenAI client is reused across calls, not recreated each time."""

    def test_openai_client_reused_across_calls(self):
        """After first call, subsequent calls reuse the same client instance."""
        from bibilab.pipeline._shared import _call_llm

        cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response"))]
        )

        with patch.object(openai, "OpenAI", return_value=mock_client_instance) as mock_openai_cls:
            # First call
            result1 = _call_llm("prompt 1", cfg)
            # Second call
            result2 = _call_llm("prompt 2", cfg)

        # Client should be created only once and reused
        assert mock_openai_cls.call_count == 1, f"Expected 1 client creation, got {mock_openai_cls.call_count}"
        # Both calls should use the same client instance
        assert result1 == "test response"
        assert result2 == "test response"
