"""Tests for stream_with_tools loop and content-block passthrough."""

from unittest.mock import MagicMock, patch

import pytest

from bibilab.pipeline._shared import StreamEvent, ToolCall, stream_llm
from tests import an_async_generator


@pytest.mark.asyncio
async def test_stream_llm_passes_list_content_to_anthropic():
    """Anthropic path: messages with list-type content are passed through as-is."""
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="anthropic", model="claude", api_key="test", base_url="")
    captured = []

    class FakeStream:
        def __init__(self, **kwargs):
            captured.append(kwargs["messages"])

        async def __aenter__(self):
            return an_async_generator([])

        async def __aexit__(self, *args):
            pass

    with patch("bibilab.pipeline._shared.AsyncAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock(messages=MagicMock(stream=FakeStream))
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "retrieve",
                        "input": {"query": "test", "search_mode": "factual"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"result":"ok"}'}]},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg, llm_max_tokens=512):
            pass

    assert len(captured) == 1
    assert isinstance(captured[0][1]["content"], list)


@pytest.mark.asyncio
async def test_stream_llm_passes_openai_tool_messages():
    """OpenAI path: tool_calls and role=tool messages pass through as-is."""
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")
    captured = []

    async def fake_create(**kwargs):
        captured.append(kwargs["messages"])
        return an_async_generator([])

    with patch("bibilab.pipeline._shared.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock(chat=MagicMock(completions=MagicMock(create=fake_create)))
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "retrieve", "arguments": '{"query":"test","search_mode":"factual"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": '{"result":"ok"}'},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg, llm_max_tokens=512):
            pass

    assert len(captured) == 1
    msgs = captured[0]
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_call_id"] == "t1"


@pytest.mark.asyncio
async def test_stream_with_tools_passthrough_no_tool_calls():
    """When LLM returns no tool_calls, events pass through and function returns."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="delta", content="Hello")
        yield StreamEvent(type="done")

    async def noop(*args, **kwargs):
        return {}

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[],
            execute_tool_fn=noop,
        ):
            events.append(event)

    assert len(events) == 2
    assert events[0].type == "delta"
    assert events[0].content == "Hello"
    assert events[1].type == "done"


@pytest.mark.asyncio
async def test_stream_with_tools_loopback_retrieve():
    """LLM calls retrieve -> execute -> feed result -> second LLM turn."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    retrieve_tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "search_mode": "factual"})
    retrieve_result = {
        "search_mode": "factual",
        "candidates_evaluated": 5,
        "sources_with_hits": 2,
        "sources_total": 3,
        "source_coverage": [],
        "_chunks": [],
    }

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=retrieve_tc)
        else:
            yield StreamEvent(type="delta", content="Answer based on chunks")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments, **kwargs):
        if tool_name == "retrieve":
            return retrieve_result
        raise ValueError(f"Unknown tool: {tool_name}")

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "what is this about?"}],
            cfg=cfg,
            tools=[RETRIEVE_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    assert call_count == 2
    tool_result_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_result_events) == 1
    delta_events = [e for e in events if e.type == "delta"]
    assert len(delta_events) == 1
    assert delta_events[0].content == "Answer based on chunks"


@pytest.mark.asyncio
async def test_stream_with_tools_terminal_tool_exits_after_execution():
    """Terminal tool (not in LOOPBACK_TOOLS): execute, yield result, exit loop."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import ToolDefinition
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    TERMINAL_TOOL = ToolDefinition(
        name="generate_report",
        description="Generate a report",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    report_tc = ToolCall(id="c1", name="generate_report", arguments={"type": "brief", "prompt": "summarize"})
    report_result = {"artifact_id": "art-1", "name": "brief", "type": "brief"}

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="delta", content="Let me generate a report.")
            yield StreamEvent(type="tool_call", tool_call=report_tc)
        else:
            yield StreamEvent(type="delta", content="Should not reach here")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments, **kwargs):
        if tool_name == "generate_report":
            return report_result
        raise ValueError(f"Unknown tool: {tool_name}")

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "make a report"}],
            cfg=cfg,
            tools=[TERMINAL_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    # Terminal tool → only one LLM call, no loopback.
    assert call_count == 1
    tool_result_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_result_events) == 1
    delta_events = [e for e in events if e.type == "delta"]
    assert len(delta_events) == 1
    assert delta_events[0].content == "Let me generate a report."
    # No done event — terminal tool suppresses it (caller adds one post-loop).
    assert not [e for e in events if e.type == "done"]
    # The tool_result carries the correct result.
    assert "art-1" in tool_result_events[0].content


@pytest.mark.asyncio
async def test_stream_with_tools_max_iterations():
    """Hard cap at MAX_TOOL_ITERATIONS prevents infinite loops."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "search_mode": "factual"})

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="tool_call", tool_call=tc)

    async def fake_execute(name, args, **kwargs):
        return {"ok": True}

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[RETRIEVE_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "Max tool iterations" in error_events[0].content


@pytest.mark.asyncio
async def test_stream_with_tools_loops_back_for_query_list_metadata():
    """A query_list_metadata tool_call must trigger a second LLM turn (loopback)."""
    from unittest.mock import AsyncMock

    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.routers.chat import stream_with_tools

    turns = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None):
        turns.append(list(messages))
        if len(turns) == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id="t1",
                    name="query_list_metadata",
                    arguments={"query_type": "count"},
                ),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="There are 8 videos.")
            yield StreamEvent(type="done")

    execute_tool_fn = AsyncMock(return_value={"count": 8})

    cfg = AIConfig(protocol="openai", model="test", api_key="test", base_url="")
    events = []

    from unittest.mock import patch

    with patch("bibilab.routers.chat.stream_llm", fake_stream_llm):
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "how many videos?"}],
            cfg=cfg,
            tools=[],
            execute_tool_fn=execute_tool_fn,
        ):
            events.append(event)

    # Two turns: initial + loopback.
    assert len(turns) == 2
    # execute_tool_fn called once with the metadata tool args.
    execute_tool_fn.assert_awaited_once_with("query_list_metadata", {"query_type": "count"}, registry={}, source_map={})
    # A delta carrying the answer must have been yielded after loopback.
    delta_text = "".join(e.content or "" for e in events if e.type == "delta")
    assert "8" in delta_text
