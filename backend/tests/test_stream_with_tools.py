"""Tests for stream_with_tools loop and content-block passthrough."""

from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
import pytest

from bibilab.pipeline._shared import (
    LLMEmptyResponseError,
    LLMOutputBudgetExceededError,
    StreamEvent,
    ToolCall,
    stream_llm,
)
from tests import an_async_generator


async def _noop_execute(*_args, **_kwargs):
    return {"_chunks": "x"}


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
                        "input": {"query": "test"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"result":"ok"}'}]},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg):
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
                        "function": {"name": "retrieve", "arguments": '{"query":"test"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": '{"result":"ok"}'},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg):
            pass

    assert len(captured) == 1
    msgs = captured[0]
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_call_id"] == "t1"


@pytest.mark.asyncio
async def test_stream_with_tools_passthrough_no_tool_calls(mock_stream_llm):
    """When LLM returns no tool_calls, events pass through and function returns."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    async def fake_stream(messages, cfg, tools=None, system=None):
        yield StreamEvent(type="delta", content="Hello")
        yield StreamEvent(type="done")

    async def noop(*args, **kwargs):
        return {}

    mock_stream_llm.side_effect = fake_stream
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
async def test_stream_with_tools_no_text_length_cutoff_raises_budget_error(mock_stream_llm):
    """No text + a length cutoff (done.stop_reason in the length set) → budget
    error, so the frontend tells the user to raise max output tokens. Pins the
    stop_reason-based branch added to de-coarsen the no-text error."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    async def fake_stream_length(messages, cfg, tools=None, system=None):
        # Thinking ate the whole budget — cut off at the token limit, no text.
        yield StreamEvent(type="done", stop_reason="length")

    async def noop(*args, **kwargs):
        return {}

    mock_stream_llm.side_effect = fake_stream_length
    with pytest.raises(LLMOutputBudgetExceededError, match="output token limit"):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[],
            execute_tool_fn=noop,
        ):
            pass


@pytest.mark.asyncio
async def test_stream_with_tools_no_text_normal_stop_raises_empty_error(mock_stream_llm):
    """No text but NOT a length cutoff (normal stop / unknown reason) →
    LLMEmptyResponseError, not a budget error: we must not give false
    'raise max output tokens' advice for a refusal or transient blank."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import LLMEmptyResponseError
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    async def fake_stream_empty(messages, cfg, tools=None, system=None):
        # Stream ended normally with no text (e.g. a refusal that emitted nothing).
        yield StreamEvent(type="done", stop_reason="stop")

    async def noop(*args, **kwargs):
        return {}

    mock_stream_llm.side_effect = fake_stream_empty
    with pytest.raises(LLMEmptyResponseError, match="no text content"):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[],
            execute_tool_fn=noop,
        ):
            pass


@pytest.mark.asyncio
async def test_stream_with_tools_loopback_find_passages(mock_stream_llm):
    """LLM calls find_passages -> execute -> feed result -> second LLM turn."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "test"})
    find_result = {
        "query": "test",
        "tool_name": FIND_PASSAGES_TOOL.name,
        "candidates_evaluated": 5,
        "sources_with_hits": 2,
        "sources_total": 3,
        "source_coverage": [],
        "_chunks": [],
        "_turn_indices": [],
        "_raw_chunks": [],
    }

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="delta", content="Answer based on chunks")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments, **kwargs):
        if tool_name == FIND_PASSAGES_TOOL.name:
            return find_result
        raise ValueError(f"Unknown tool: {tool_name}")

    mock_stream_llm.side_effect = fake_stream
    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "what is this about?"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
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
async def test_preamble_trigger_attached_per_decision_point(mock_stream_llm):
    """Trigger is persisted at the initial question AND each tool result (one per decision point, no accumulation).
    The round's preamble text lands in the assistant message's content so the prompt-trace dump shows it."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import _build_preamble_trigger, stream_with_tools

    trigger = _build_preamble_trigger("en")
    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})
    call_count = 0

    async def stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="delta", content="looking that up.")
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="delta", content="Answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = stream
    sink: list[dict] = []
    async for _ in stream_with_tools(
        messages=[{"role": "user", "content": "what is this about?"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
        execute_tool_fn=_noop_execute,
        messages_sink=sink,
    ):
        pass

    # initial question: trigger folded into the question string, not a second user turn
    assert sink[0] == {"role": "user", "content": f"what is this about?\n\n{trigger}"}
    # tool-result decision point: trigger appended after the tool message (tool→user, not user→user)
    tool_idx = next(i for i, m in enumerate(sink) if m.get("role") == "tool")
    assert sink[tool_idx + 1] == {"role": "user", "content": trigger}
    # one trigger per decision point (initial + one tool result), no accumulation
    decision_points = [m for m in sink if isinstance(m.get("content"), str) and m["content"].endswith(trigger)]
    assert len(decision_points) == 2, sink
    # preamble text emitted before the tool_call lives on the assistant message (dump-visible)
    asst_with_tool = next(m for m in sink if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst_with_tool["content"] == "looking that up."


@pytest.mark.asyncio
async def test_preamble_trigger_folds_into_anthropic_tool_result_round(mock_stream_llm):
    """Anthropic: trigger folds into the trailing user message (round 1 question; round 2 tool_result)."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import _build_preamble_trigger, stream_with_tools

    trigger = _build_preamble_trigger("en")
    cfg = AIConfig(protocol="anthropic", model="claude", api_key="test", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})
    call_count = 0

    async def stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="delta", content="Answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = stream
    sink: list[dict] = []
    async for _ in stream_with_tools(
        messages=[{"role": "user", "content": "what is this about?"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
        execute_tool_fn=_noop_execute,
        messages_sink=sink,
    ):
        pass

    # round 1: trigger folded into the original user message's content blocks
    user_msgs = [m for m in sink if m.get("role") == "user" and "what is this about?" in str(m.get("content"))]
    assert {"type": "text", "text": trigger} in user_msgs[0]["content"]

    # round 2: trigger folded into the trailing tool_result user message
    tool_result_msgs = [
        m
        for m in sink
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" and b.get("tool_use_id") == "c1" for b in m["content"])
    ]
    assert {"type": "text", "text": trigger} in tool_result_msgs[0]["content"]


@pytest.mark.asyncio
async def test_whitespace_only_preamble_emits_no_text_block(mock_stream_llm):
    """A whitespace-only delta before a tool call must NOT become a text block on the
    fed-back assistant message — Anthropic rejects whitespace-only text blocks, and the
    OpenAI content must be None, not a blank string."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import stream_with_tools

    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})

    for protocol in ("anthropic", "openai"):
        call_count = 0

        async def stream(messages, cfg, tools=None, system=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamEvent(type="delta", content="\n  ")  # stray whitespace only
                yield StreamEvent(type="tool_call", tool_call=find_tc)
            else:
                yield StreamEvent(type="delta", content="Answer")
                yield StreamEvent(type="done")

        cfg = AIConfig(protocol=protocol, model="m", api_key="test", base_url="")
        mock_stream_llm.side_effect = stream
        sink: list[dict] = []
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[FIND_PASSAGES_TOOL],
            execute_tool_fn=_noop_execute,
            messages_sink=sink,
        ):
            pass

        if protocol == "anthropic":
            asst = next(
                m
                for m in sink
                if m.get("role") == "assistant"
                and isinstance(m.get("content"), list)
                and any(b.get("type") == "tool_use" for b in m["content"])
            )
            assert all(b.get("type") != "text" for b in asst["content"]), asst
        else:
            asst = next(m for m in sink if m.get("role") == "assistant" and m.get("tool_calls"))
            assert asst["content"] is None, asst


@pytest.mark.asyncio
async def test_preamble_trigger_skipped_before_forced_synthesis(mock_stream_llm):
    """Forced synthesis turn gets no trigger (synthesis directive follows the last tool result directly)."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import (
        MAX_TOOL_ITERATIONS,
        _build_preamble_trigger,
        _build_synthesis_directive,
        stream_with_tools,
    )

    trigger = _build_preamble_trigger("en")
    synthesis = _build_synthesis_directive("en")
    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")
    call_count = 0

    async def stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id=f"c{call_count}", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"}),
            )
        else:
            yield StreamEvent(type="delta", content="final answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = stream
    sink: list[dict] = []
    async for _ in stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
        execute_tool_fn=_noop_execute,
        messages_sink=sink,
    ):
        pass

    triggers = [m for m in sink if isinstance(m.get("content"), str) and m["content"].endswith(trigger)]
    assert len(triggers) == MAX_TOOL_ITERATIONS, sink
    synth_idx = next(i for i, m in enumerate(sink) if m.get("content") == synthesis)
    assert sink[synth_idx - 1].get("role") == "tool"


@pytest.mark.asyncio
async def test_stream_with_tools_max_iterations_graceful(mock_stream_llm):
    """After MAX_TOOL_ITERATIONS the loop stops *executing* tools but keeps them
    advertised, so the serving layer's tool-call grammar stays on and a stubborn
    tool attempt parses as an ignored structured tool_call instead of leaking
    native tool-call tokens as the answer."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "test"})

    iterations_seen = []

    async def fake_stream(messages, cfg, tools=None, system=None):
        # Record how many tools were available in this call
        iterations_seen.append(len(tools) if tools else 0)

        # ALWAYS yield a tool call — the iteration limit is tracked INSIDE stream_with_tools
        # We need to keep returning tools so we hit the iteration > MAX_TOOL_ITERATIONS check
        yield StreamEvent(type="tool_call", tool_call=tc)
        # Note: we don't yield done here — we rely on stream_with_tools to stop the loop
        # after MAX_TOOL_ITERATIONS

    async def fake_execute(name, args, **kwargs):
        return {"ok": True, "_chunks": "stub"}

    mock_stream_llm.side_effect = fake_stream
    events = []
    # The mock never yields a done event, so last_stop_reason stays None (not a
    # length cutoff) → empty-response error, not a budget error.
    with pytest.raises(LLMEmptyResponseError, match="no text content"):
        async for event in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[FIND_PASSAGES_TOOL],
            execute_tool_fn=fake_execute,
        ):
            events.append(event)

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 0, (
        f"No hard error event — the typed no-text exception surfaces the "
        f"no-text condition. Got error events: {error_events}"
    )

    # MAX_TOOL_ITERATIONS tool-using turns + 1 synthesis turn + 1 forced synthesis
    # (forced synthesis kicks in because the mock never produces text).
    assert len(iterations_seen) == MAX_TOOL_ITERATIONS + 2
    # Tools stay advertised on the synthesis turn and forced synthesis (grammar on,
    # anti-leak); they are simply never executed past MAX_TOOL_ITERATIONS.
    assert iterations_seen[-2] == 1
    assert iterations_seen[-1] == 1


@pytest.mark.asyncio
async def test_stream_with_tools_forces_synthesis_when_exhausted_turn_returns_empty(mock_stream_llm):
    """When the synthesis turn produces no text, force one more LLM call so the user always gets an answer."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "test"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            # Tool iterations: always call find_passages
            yield StreamEvent(type="tool_call", tool_call=find_tc)
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
            "_chunks": "stub",
        }

    mock_stream_llm.side_effect = fake_stream
    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "what is the answer?"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
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
async def test_forced_synthesis_forwards_error_events(mock_stream_llm):
    """If the forced-synthesis LLM call yields an error event, it must be forwarded
    so run_chat_turn can mark the message failed instead of silently dropping it."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
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
            "_chunks": "stub",
        }

    mock_stream_llm.side_effect = fake_stream
    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
        execute_tool_fn=fake_execute,
    ):
        events.append(event)

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "forced synthesis failed" in error_events[0].content


@pytest.mark.asyncio
async def test_stream_with_tools_populates_tool_block_sink(mock_stream_llm):
    """tool_block_sink collects normalized entries for each tool call executed."""
    from bibilab.config import AIConfig
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "test"})
    find_result = {
        "query": "test",
        "tool_name": FIND_PASSAGES_TOOL.name,
        "candidates_evaluated": 5,
        "sources_with_hits": 1,
        "sources_total": 1,
        "source_coverage": [{"source_id": "s1", "title": "V1"}],
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

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="delta", content="Answer")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments, **kwargs):
        return find_result

    sink: list = []
    mock_stream_llm.side_effect = fake_stream
    async for _ in stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=cfg,
        tools=[FIND_PASSAGES_TOOL],
        execute_tool_fn=fake_execute,
        tool_block_sink=sink,
    ):
        pass

    assert len(sink) == 1
    entry = sink[0]
    assert entry["tool_use_id"] == "c1"
    assert entry["name"] == FIND_PASSAGES_TOOL.name
    assert entry["arguments"] == {"query": "test"}
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
        from bibilab.pipeline._shared import _classify_llm_error

        assert _classify_llm_error(Exception("something broke")) == "internal_error"

    def test_context_window_exceeded(self):
        """resolve_max_tokens' overflow error gets its own code, not internal_error,
        so the chat toast is meaningful rather than a generic server error."""
        from bibilab.pipeline._shared import ContextWindowExceededError, _classify_llm_error

        assert _classify_llm_error(ContextWindowExceededError("too big")) == "llm_context_window_exceeded"

    def test_openai_connection_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert _classify_llm_error(openai.APIConnectionError(request=self._req)) == "llm_connection_error"

    def test_openai_timeout(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert _classify_llm_error(openai.APITimeoutError(request=self._req)) == "llm_connection_error"

    def test_openai_auth_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(openai.AuthenticationError(message="bad key", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_openai_permission_denied(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(openai.PermissionDeniedError(message="not allowed", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_openai_rate_limit(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(openai.RateLimitError(message="too many", response=self._resp, body=None))
            == "llm_rate_limit_error"
        )

    def test_openai_api_error_subclass(self):
        from bibilab.pipeline._shared import _classify_llm_error

        class SomeOpenAIError(openai.APIError):
            pass

        assert _classify_llm_error(SomeOpenAIError(message="generic", request=self._req, body=None)) == "llm_api_error"

    def test_anthropic_connection_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert _classify_llm_error(anthropic.APIConnectionError(request=self._req)) == "llm_connection_error"

    def test_anthropic_auth_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(anthropic.AuthenticationError(message="bad key", response=self._resp, body=None))
            == "llm_auth_error"
        )

    def test_anthropic_rate_limit(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(anthropic.RateLimitError(message="too many", response=self._resp, body=None))
            == "llm_rate_limit_error"
        )

    def test_openai_api_status_error_still_api_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(openai.APIStatusError(message="status 500", response=self._resp, body=None))
            == "llm_api_error"
        )

    def test_anthropic_timeout(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert _classify_llm_error(anthropic.APITimeoutError(request=self._req)) == "llm_connection_error"

    def test_anthropic_api_status_error_still_api_error(self):
        from bibilab.pipeline._shared import _classify_llm_error

        assert (
            _classify_llm_error(anthropic.APIStatusError(message="status 500", response=self._resp, body=None))
            == "llm_api_error"
        )


@pytest.mark.asyncio
async def test_stream_with_tools_default_sink_none_does_not_crash(mock_stream_llm):
    """Calling stream_with_tools without tool_block_sink must not crash (defaults to None)."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    async def fake_stream(messages, cfg, tools=None, system=None):
        yield StreamEvent(type="delta", content="Hello")
        yield StreamEvent(type="done")

    async def noop(*args, **kwargs):
        return {}

    mock_stream_llm.side_effect = fake_stream
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


@pytest.mark.asyncio
async def test_second_retrieval_allowed_no_sequential_guard(mock_stream_llm):
    """v2: find_passages then read_section in successive iterations is allowed.
    The sequential guard is deleted; multi-hop is bounded only by the iter cap."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL, READ_SECTION_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url="")

    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "test"})
    read_tc = ToolCall(id="c2", name=READ_SECTION_TOOL.name, arguments={"section_id": "[1]"})

    calls: list[str] = []
    iter_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, **kwargs):
        nonlocal iter_count
        iter_count += 1
        if iter_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        elif iter_count == 2:
            yield StreamEvent(type="tool_call", tool_call=read_tc)
        else:
            yield StreamEvent(type="delta", content="Answer.")
            yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        calls.append(name)
        return {"_chunks": "x", "section_id": "sec-1", "source_id": "s1"}

    mock_stream_llm.side_effect = fake_stream_llm
    evs = [
        ev
        async for ev in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[FIND_PASSAGES_TOOL, READ_SECTION_TOOL],
            execute_tool_fn=fake_execute,
        )
    ]
    assert evs, "stream must produce events"

    # The guard is DELETED: both find_passages AND read_section execute in successive iterations.
    assert calls == [FIND_PASSAGES_TOOL.name, READ_SECTION_TOOL.name]


@pytest.mark.asyncio
async def test_stream_with_tools_shares_one_seen_set_across_calls():
    """The turn-scoped seen_chunk_ids set is created once and passed to every
    execute_tool call so parallel/multi-hop find_passages share dedup state."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    seen_sets = []

    async def fake_execute(name, args, **kwargs):
        seen_sets.append(kwargs.get("seen_chunk_ids"))
        return {"_chunks": "ok", "tool_name": name}

    # First stream yields two parallel tool calls; second yields a plain answer.
    streams = [
        [
            StreamEvent(type="tool_call", tool_call=ToolCall(id="a", name="find_passages", arguments={"query": "x"})),
            StreamEvent(type="tool_call", tool_call=ToolCall(id="b", name="find_passages", arguments={"query": "y"})),
            StreamEvent(type="done"),
        ],
        [StreamEvent(type="delta", content="answer"), StreamEvent(type="done")],
    ]

    async def fake_stream_llm(*args, **kwargs):
        for ev in streams.pop(0):
            yield ev

    cfg = AIConfig(protocol="openai", model="m", api_key="k", base_url="")
    with patch("bibilab.routers.chat.stream_llm", fake_stream_llm):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
            tools=[],
            execute_tool_fn=fake_execute,
        ):
            pass

    assert len(seen_sets) == 2  # two parallel find_passages calls
    assert seen_sets[0] is seen_sets[1]  # same set object → shared turn state
    assert seen_sets[0] is not None
