"""Tests for LLM streaming and tool calling abstraction."""

from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest


class TestToolDefinitionFormatTranslation:
    """ToolDefinition translates correctly to protocol-specific formats."""

    def test_tool_definition_to_openai_format(self):
        from bibilab.pipeline._shared import ToolDefinition, _to_openai_tool

        tool = ToolDefinition(
            name="generate_report",
            description="Generate a report from sources.",
            parameters={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["brief", "study_guide"],
                    },
                },
                "required": ["report_type"],
            },
        )
        openai_fmt = _to_openai_tool(tool)
        assert openai_fmt == {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "Generate a report from sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_type": {
                            "type": "string",
                            "enum": ["brief", "study_guide"],
                        },
                    },
                    "required": ["report_type"],
                },
            },
        }

    def test_tool_definition_to_anthropic_format(self):
        from bibilab.pipeline._shared import ToolDefinition, _to_anthropic_tool

        tool = ToolDefinition(
            name="generate_report",
            description="Generate a report from sources.",
            parameters={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["brief", "study_guide"],
                    },
                },
                "required": ["report_type"],
            },
        )
        anthropic_fmt = _to_anthropic_tool(tool)
        assert anthropic_fmt == {
            "name": "generate_report",
            "description": "Generate a report from sources.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["brief", "study_guide"],
                    },
                },
                "required": ["report_type"],
            },
        }


class TestOpenAIStreaming:
    """OpenAI streaming yields text deltas incrementally and buffers tool calls."""

    @pytest.mark.asyncio
    async def test_stream_llm_openai_yields_text_deltas(self, tmp_bibilab_home):
        from bibilab.config import AIConfig
        from bibilab.pipeline._shared import stream_llm

        cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test-openai-deltas",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello ", tool_calls=None), index=0)]

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content="world", tool_calls=None), index=0)]

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock(delta=MagicMock(content=".", tool_calls=None), index=0)]

        mock_response = MagicMock()
        mock_response.__iter__ = lambda self: iter([mock_chunk1, mock_chunk2, mock_chunk3])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(openai, "OpenAI", return_value=mock_client):
            events = [e async for e in stream_llm([{"role": "user", "content": "hi"}], cfg)]

        deltas = [e for e in events if e.type == "delta"]
        assert len(deltas) == 3
        assert deltas[0].content == "Hello "
        assert deltas[1].content == "world"
        assert deltas[2].content == "."

    @pytest.mark.asyncio
    async def test_stream_llm_openai_buffers_tool_call_arguments(self, tmp_bibilab_home):
        from bibilab.config import AIConfig
        from bibilab.pipeline._shared import stream_llm

        cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test-openai-toolcall",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )

        mock_text_chunk = MagicMock()
        mock_text_chunk.choices = [MagicMock(delta=MagicMock(content="I will ", tool_calls=None), index=0)]

        mock_tool_delta1 = MagicMock()
        mock_function1 = MagicMock()
        mock_function1.name = "generate_report"
        mock_function1.arguments = '{"type":'
        mock_tool_delta1.choices = [
            MagicMock(
                delta=MagicMock(
                    content=None,
                    tool_calls=[
                        MagicMock(index=0, id="call_1", function=mock_function1),
                    ],
                ),
                index=0,
            )
        ]

        mock_tool_delta2 = MagicMock()
        mock_function2 = MagicMock()
        mock_function2.name = None
        mock_function2.arguments = ' "study_guide"}'
        mock_tool_delta2.choices = [
            MagicMock(
                delta=MagicMock(
                    content=None,
                    tool_calls=[
                        MagicMock(index=0, id="call_1", function=mock_function2),
                    ],
                ),
                index=0,
            )
        ]

        mock_response = MagicMock()
        mock_response.__iter__ = lambda self: iter([mock_text_chunk, mock_tool_delta1, mock_tool_delta2])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        tool_def = MagicMock()
        tool_def.name = "generate_report"

        with patch.object(openai, "OpenAI", return_value=mock_client):
            events = [e async for e in stream_llm([{"role": "user", "content": "hi"}], cfg, tools=[tool_def])]

        deltas = [e for e in events if e.type == "delta"]
        assert deltas[0].content == "I will "

        tool_calls = [e for e in events if e.type == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "generate_report"
        assert tool_calls[0].tool_call.arguments == {"type": "study_guide"}


class TestAnthropicStreaming:
    """Anthropic streaming yields text deltas incrementally and emits tool_call events."""

    @pytest.mark.asyncio
    async def test_stream_llm_anthropic_yields_text_deltas(self, tmp_bibilab_home):
        from bibilab.config import AIConfig
        from bibilab.pipeline._shared import stream_llm

        cfg = AIConfig(
            protocol="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-deltas",
            base_url="",
            output_language="en",
        )

        mock_text_event1 = MagicMock()
        mock_text_event1.type = "content_block_delta"
        mock_text_event1.delta = MagicMock(type="text_delta", text="Hello ")

        mock_text_event2 = MagicMock()
        mock_text_event2.type = "content_block_delta"
        mock_text_event2.delta = MagicMock(type="text_delta", text="world")

        mock_message_event = MagicMock()
        mock_message_event.type = "message_stop"

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([mock_text_event1, mock_text_event2, mock_message_event])

        mock_client = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_stream)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_client.messages.stream.return_value = mock_cm

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            events = [e async for e in stream_llm([{"role": "user", "content": "hi"}], cfg)]

        deltas = [e for e in events if e.type == "delta"]
        assert len(deltas) == 2
        assert deltas[0].content == "Hello "
        assert deltas[1].content == "world"

    @pytest.mark.asyncio
    async def test_stream_llm_anthropic_yields_tool_call_on_content_block_stop(self, tmp_bibilab_home):
        from bibilab.config import AIConfig
        from bibilab.pipeline._shared import stream_llm

        cfg = AIConfig(
            protocol="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-toolcall",
            base_url="",
            output_language="en",
        )

        mock_text_event = MagicMock()
        mock_text_event.type = "content_block_delta"
        mock_text_event.delta = MagicMock(type="text_delta", text="Generating report...")

        mock_tool_event = MagicMock()
        mock_tool_event.type = "content_block_stop"
        mock_content_block = MagicMock()
        mock_content_block.type = "tool_use"
        mock_content_block.id = "tool_use_1"
        mock_content_block.name = "generate_report"
        mock_content_block.input = {"type": "study_guide", "prompt": "summarize"}
        mock_tool_event.content_block = mock_content_block

        mock_message_event = MagicMock()
        mock_message_event.type = "message_stop"

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([mock_text_event, mock_tool_event, mock_message_event])

        mock_client = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_stream)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_client.messages.stream.return_value = mock_cm

        tool_def = MagicMock()
        tool_def.name = "generate_report"

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            events = [e async for e in stream_llm([{"role": "user", "content": "hi"}], cfg, tools=[tool_def])]

        deltas = [e for e in events if e.type == "delta"]
        assert deltas[0].content == "Generating report..."

        tool_calls = [e for e in events if e.type == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "generate_report"
        assert tool_calls[0].tool_call.arguments == {"type": "study_guide", "prompt": "summarize"}


class TestStreamLlMErrorHandling:
    """stream_llm raises on unrecoverable errors."""

    @pytest.mark.asyncio
    async def test_stream_llm_openai_raises_on_error(self, tmp_bibilab_home):
        from bibilab.config import AIConfig
        from bibilab.pipeline._shared import stream_llm

        cfg = AIConfig(
            protocol="openai",
            model="gpt-4o-mini",
            api_key="sk-test-openai-error",
            base_url="https://api.openai.com/v1",
            output_language="en",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="rate limited", request=MagicMock(), body=None
        )

        with patch.object(openai, "OpenAI", return_value=mock_client):
            with pytest.raises(openai.APIError):
                [e async for e in stream_llm([{"role": "user", "content": "hi"}], cfg)]
