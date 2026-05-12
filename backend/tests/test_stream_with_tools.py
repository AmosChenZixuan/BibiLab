"""Tests for stream_with_tools loop and content-block passthrough."""

from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
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
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

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
                        "input": {"query": "test", "expected_hits": "few"},
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
                        "function": {"name": "retrieve", "arguments": '{"query":"test","expected_hits":"few"}'},
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

    # done is no longer yielded by stream_with_tools — caller emits it post-loop.
    assert len(events) == 1
    assert events[0].type == "delta"
    assert events[0].content == "Hello"


@pytest.mark.asyncio
async def test_stream_with_tools_loopback_retrieve():
    """LLM calls retrieve -> execute -> feed result -> second LLM turn."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    retrieve_tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "expected_hits": "few"})
    retrieve_result = {
        "expected_hits": "few",
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
async def test_stream_with_tools_max_iterations_graceful():
    """After MAX_TOOL_ITERATIONS, active_tools is forced to [] so the LLM synthesizes from accumulated results."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "expected_hits": "few"})

    iterations_seen = []

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        # Record how many tools were available in this call
        iterations_seen.append(len(tools) if tools else 0)

        # ALWAYS yield a tool call — the iteration limit is tracked INSIDE stream_with_tools
        # We need to keep returning tools so we hit the iteration > MAX_TOOL_ITERATIONS check
        yield StreamEvent(type="tool_call", tool_call=tc)
        # Note: we don't yield done here — we rely on stream_with_tools to stop the loop
        # after MAX_TOOL_ITERATIONS

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
    assert len(error_events) == 0, f"No hard error — forced synthesis instead. Got error events: {error_events}"

    # MAX_TOOL_ITERATIONS tool-using turns + 1 synthesis turn + 1 forced synthesis
    # (forced synthesis kicks in because the mock never produces text).
    assert len(iterations_seen) == MAX_TOOL_ITERATIONS + 2
    # Both the synthesis turn and forced synthesis have no tools available.
    assert iterations_seen[-2] == 0
    assert iterations_seen[-1] == 0


@pytest.mark.asyncio
async def test_stream_with_tools_forces_synthesis_when_exhausted_turn_returns_empty():
    """When the synthesis turn produces no text, force one more LLM call so the user always gets an answer."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    retrieve_tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "expected_hits": "few"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            # Tool iterations: always call retrieve
            yield StreamEvent(type="tool_call", tool_call=retrieve_tc)
        elif call_count == MAX_TOOL_ITERATIONS + 1:
            # Synthesis turn: model returns empty — no deltas, no tool calls
            yield StreamEvent(type="done")
        else:
            # Forced synthesis: model produces the answer
            yield StreamEvent(type="delta", content="Based on retrieved sources, the answer is 42.")
            yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        return {
            "ok": True,
            "query": args.get("query", ""),
            "source_coverage": [],
            "candidates_evaluated": 1,
            "sources_with_hits": 0,
            "sources_total": 1,
        }

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "what is the answer?"}],
            cfg=cfg,
            tools=[RETRIEVE_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    # MAX_TOOL_ITERATIONS tool turns + 1 empty synthesis + 1 forced synthesis
    assert call_count == MAX_TOOL_ITERATIONS + 2
    # No errors
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 0
    # The forced synthesis produced text
    delta_text = "".join(e.content or "" for e in events if e.type == "delta")
    assert "42" in delta_text


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
    execute_tool_fn.assert_awaited_once()
    call_args = execute_tool_fn.call_args
    assert call_args[0][0] == "query_list_metadata"
    assert call_args[0][1] == {"query_type": "count"}
    # A delta carrying the answer must have been yielded after loopback.
    delta_text = "".join(e.content or "" for e in events if e.type == "delta")
    assert "8" in delta_text


@pytest.mark.asyncio
async def test_forced_synthesis_forwards_error_events():
    """If the forced-synthesis LLM call yields an error event, it must be forwarded
    so run_chat_turn can mark the message failed instead of silently dropping it."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")
    retrieve_tc = ToolCall(id="c1", name="retrieve", arguments={"query": "q", "expected_hits": "few"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="tool_call", tool_call=retrieve_tc)
        elif call_count == MAX_TOOL_ITERATIONS + 1:
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="error", content="forced synthesis failed")

    async def fake_execute(name, args, **kwargs):
        return {
            "ok": True,
            "query": "q",
            "source_coverage": [],
            "candidates_evaluated": 1,
            "sources_with_hits": 0,
            "sources_total": 1,
        }

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        events = []
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[RETRIEVE_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "forced synthesis failed" in error_events[0].content


@pytest.mark.asyncio
async def test_stream_with_tools_populates_tool_block_sink():
    """tool_block_sink collects normalized entries for each tool call executed."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    retrieve_tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "expected_hits": "few"})
    retrieve_result = {
        "query": "test",
        "expected_hits": "few",
        "candidates_evaluated": 5,
        "sources_with_hits": 1,
        "sources_total": 1,
        "source_coverage": [{"source_id": "s1", "video_id": "v1", "title": "V1"}],
        "_chunks": "internal",
        "_turn_indices": [1],
        "_raw_chunks": [
            {
                "source_id": "s1",
                "chunk_id": "v1_120_145",
                "content": "verbatim",
                "video_title": "V1",
                "timestamp_start": 120.0,
                "timestamp_end": 145.0,
                "citation_index": 1,
            },
        ],
    }

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=retrieve_tc)
        else:
            yield StreamEvent(type="delta", content="Answer")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments, **kwargs):
        return retrieve_result

    sink: list = []
    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[RETRIEVE_TOOL],
            execute_tool_fn=fake_execute,
            tool_block_sink=sink,
        ):
            pass

    assert len(sink) == 1
    entry = sink[0]
    assert entry["tool_use_id"] == "c1"
    assert entry["name"] == "retrieve"
    assert entry["arguments"] == {"query": "test", "expected_hits": "few"}
    assert "_chunks" not in entry["result"]
    assert "_raw_chunks" not in entry["result"]
    assert entry["result"]["chunks"][0]["content"] == "verbatim"


class TestClassifyError:
    _req = httpx.Request("GET", "http://test")
    _resp = httpx.Response(500, request=_req)

    def test_tool_error_not_classified_by_sdk(self):
        """A plain Exception (like tool execution failures) is internal_error.

        The tool_error code is set explicitly at the yield site in run_chat_turn,
        not derived from exception inspection.
        """
        from bibilab.routers.chat import classify_error

        assert classify_error(Exception("something broke")) == "internal_error"

    def test_openai_connection_error(self):
        from bibilab.routers.chat import classify_error

        assert classify_error(openai.APIConnectionError(request=self._req)) == "llm_connection_error"

    def test_openai_timeout(self):
        from bibilab.routers.chat import classify_error

        assert classify_error(openai.APITimeoutError(request=self._req)) == "llm_connection_error"

    def test_openai_auth_error(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(openai.AuthenticationError(message="bad key", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_openai_permission_denied(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(openai.PermissionDeniedError(message="not allowed", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_openai_rate_limit(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(openai.RateLimitError(message="too many", response=self._resp, body=None))
            == "llm_rate_limit_error"
        )

    def test_openai_api_error_subclass(self):
        from bibilab.routers.chat import classify_error

        class SomeOpenAIError(openai.APIError):
            pass

        assert classify_error(SomeOpenAIError(message="generic", request=self._req, body=None)) == "llm_api_error"

    def test_anthropic_connection_error(self):
        from bibilab.routers.chat import classify_error

        assert classify_error(anthropic.APIConnectionError(request=self._req)) == "llm_connection_error"

    def test_anthropic_auth_error(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(anthropic.AuthenticationError(message="bad key", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_anthropic_rate_limit(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(anthropic.RateLimitError(message="too many", response=self._resp, body=None))
            == "llm_rate_limit_error"
        )

    def test_openai_api_status_error_still_api_error(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(openai.APIStatusError(message="status 500", response=self._resp, body=None))
            == "llm_api_error"
        )

    def test_anthropic_timeout(self):
        from bibilab.routers.chat import classify_error

        assert classify_error(anthropic.APITimeoutError(request=self._req)) == "llm_connection_error"

    def test_anthropic_api_status_error_still_api_error(self):
        from bibilab.routers.chat import classify_error

        assert (
            classify_error(anthropic.APIStatusError(message="status 500", response=self._resp, body=None))
            == "llm_api_error"
        )


@pytest.mark.asyncio
async def test_stream_with_tools_default_sink_none_does_not_crash():
    """Calling stream_with_tools without tool_block_sink must not crash (defaults to None)."""
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

    assert len(events) == 1
    assert events[0].type == "delta"
    assert events[0].content == "Hello"
