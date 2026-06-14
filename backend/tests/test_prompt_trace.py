"""Tests for opt-in end-of-turn dump (input + response) — single file per chat message.

Per-turn writes (one per LLM call) were replaced by a single end-of-turn write
because the final LLM call's `messages` list is the cumulative state — it
contains all prior tool results. One file per message captures the final state
that the LLM actually saw, with N× less storage and N× less I/O.

Also covers the LLM-bound tool message content (feed `_chunks` only,
no FTS noise) and the tool_block_sink contract for raw_chunks."""

import json
import logging

import pytest

from bibilab.config import AIConfig, bibilab_home
from bibilab.pipeline._shared import StreamEvent, ToolCall, ToolDefinition
from bibilab.pipeline.chat_tools import TOOL_NAME_FIND_PASSAGES


def test_dump_turn_writes_one_file_per_message(tmp_bibilab_home):
    """Core: one file captures system, tools, messages, and the LLM response."""
    from bibilab.routers.chat import _dump_turn

    debug_dir = bibilab_home() / "debug"
    debug_dir.mkdir()
    debug_path = debug_dir / "msg_x.json"
    _dump_turn(
        debug_path,
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        response_text="answer",
    )
    payload = json.loads(debug_path.read_text())
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "q"}]
    assert payload["response"]["text"] == "answer"


def test_dump_turn_preserves_cjk(tmp_bibilab_home):
    """CJK must never be escaped to \\uXXXX in the dump."""
    from bibilab.routers.chat import _dump_turn

    debug_dir = bibilab_home() / "debug"
    debug_dir.mkdir()
    debug_path = debug_dir / "msg_x.json"
    _dump_turn(
        debug_path,
        system="你是助手",
        messages=[{"role": "user", "content": "用户提问"}],
        tools=[],
        response_text="回答如下",
    )
    raw = debug_path.read_text()
    assert "你是助手" in raw
    assert "用户提问" in raw
    assert "回答如下" in raw
    assert "\\u" not in raw


def test_dump_turn_serializes_tool_definitions(tmp_bibilab_home):
    """Tool definitions are reduced to {name, description, parameters}."""
    from bibilab.routers.chat import _dump_turn

    debug_dir = bibilab_home() / "debug"
    debug_dir.mkdir()
    debug_path = debug_dir / "msg_x.json"
    tools = [
        ToolDefinition(
            name=TOOL_NAME_FIND_PASSAGES,
            description="locator",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    _dump_turn(debug_path, system="s", messages=[], tools=tools, response_text="r")
    payload = json.loads(debug_path.read_text())
    assert payload["tools"] == [
        {"name": TOOL_NAME_FIND_PASSAGES, "description": "locator", "parameters": tools[0].parameters}
    ]


def test_dump_turn_swallows_errors(tmp_bibilab_home, caplog):
    """A write failure must be logged, not propagated."""
    from bibilab.routers import chat as chat_module

    # Point at a path that is a regular file — write_text on the full file path
    # will fail because the parent is not a directory.
    blocker = tmp_bibilab_home / "blocker"
    blocker.write_text("not a dir")
    debug_path = blocker / "msg_x.json"  # cannot be written — parent is a file
    with caplog.at_level(logging.WARNING, logger="bibilab.routers.chat"):
        chat_module._dump_turn(debug_path, system="sys", messages=[], tools=[])
    assert blocker.read_text() == "not a dir"
    assert any("dump_turn_failed" in r.message for r in caplog.records)


def test_dump_turn_includes_model_and_timestamp(tmp_bibilab_home):
    """Payload records the model name and ISO timestamp for the turn."""
    from bibilab.routers.chat import _dump_turn

    debug_dir = bibilab_home() / "debug"
    debug_dir.mkdir()
    debug_path = debug_dir / "msg_x.json"
    _dump_turn(
        debug_path,
        system="s",
        messages=[],
        tools=[],
        response_text="r",
        model="gpt-4o",
        timestamp="2026-06-06T14:23:11+08:00",
    )
    payload = json.loads(debug_path.read_text())
    assert payload["model"] == "gpt-4o"
    assert payload["timestamp"] == "2026-06-06T14:23:11+08:00"


# Representative find_passages result mirroring chat_tools.py shape:
# `_chunks` is the only LLM-facing field; everything else is noise/telemetry/bookkeeping.
_NOISY_FIND_PASSAGES_RESULT = {
    "_chunks": '[1] "sample excerpt one"\n[2] "sample excerpt two"',
    "_raw_chunks": [
        {
            "content": "测 试 文 本 不 应 进 入 大 模 型  测试 试文 文本 本文 本应 应进 进入 入大 大模 模型",
            "chunk_id": "c1",
        },
        {"content": "FTS bigram garbage should not reach the LLM", "chunk_id": "c2"},
    ],
    "source_coverage": [
        {"source_id": "s1", "source_title": "Source A"},
        {"source_id": "s2", "source_title": "Source B"},
    ],
    "candidates_evaluated": 42,
    "sources_with_hits": 7,
    "sources_total": 12,
    "reranked": 8,
    "scoped_pool_size": 12,
    "facet_scope": {"no_match": False, "matched_count": 12},
    "tool_name": TOOL_NAME_FIND_PASSAGES,
}


async def _drive_stream_with_tools(
    mock_stream_llm, *, protocol: str, tool_result: dict, tool_block_sink: list[dict] | None = None
):
    """Run stream_with_tools for one find_passages iteration; return the second
    LLM call's messages list, the SSE tool_result event content, and the
    tool_block_sink entries (if a sink was passed in — it's mutated in place,
    so callers can also read it directly from the argument they passed)."""
    from bibilab.routers import chat as chat_module

    captured: dict = {"calls": [], "sse_tool_result": None}

    async def fake_stream_llm(messages, cfg, tools, **kwargs):
        captured["calls"].append(list(messages))
        if len(captured["calls"]) == 1:
            tc = ToolCall(id="toolu_a", name=TOOL_NAME_FIND_PASSAGES, arguments={"query": "q"})
            yield StreamEvent(type="tool_call", tool_call=tc)
            yield StreamEvent(type="done")
            return
        yield StreamEvent(type="delta", content="answer")
        yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        return tool_result

    mock_stream_llm.side_effect = fake_stream_llm

    cfg = AIConfig(protocol=protocol, model="x", api_key="k", base_url="")
    gen = chat_module.stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=cfg,
        tools=[],
        execute_tool_fn=fake_execute,
        tool_block_sink=tool_block_sink,
    )
    async for ev in gen:
        if ev.type == "tool_result":
            captured["sse_tool_result"] = json.loads(ev.content)

    return {"second_call_messages": captured["calls"][-1], "sse_tool_result": captured["sse_tool_result"]}


def _last_tool_result_content(messages: list[dict], protocol: str) -> str:
    """Protocol-aware extract of the most recent tool-result content (trigger is now the tail, not the tool result)."""
    if protocol == "anthropic":
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                block = next((b for b in m["content"] if b.get("type") == "tool_result"), None)
                if block is not None:
                    return block["content"]
        raise AssertionError(f"no tool_result block found in {messages}")
    for m in reversed(messages):
        if m.get("role") == "tool":
            return m["content"]
    raise AssertionError(f"no tool message found in {messages}")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_llm_tool_message_feeds_only_chunks_excerpts_no_noise(mock_stream_llm, protocol):
    """LLM tool message content is `_chunks` only — not the full result dict.
    Asserts the symmetric Anthropic + OpenAI branches; client SSE payload is unchanged.
    """
    captured = await _drive_stream_with_tools(
        mock_stream_llm, protocol=protocol, tool_result=_NOISY_FIND_PASSAGES_RESULT
    )

    second = captured["second_call_messages"]
    content = _last_tool_result_content(second, protocol)

    expected = _NOISY_FIND_PASSAGES_RESULT["_chunks"]
    assert content == expected, f"LLM tool message must equal `_chunks` exactly; got {content!r}"

    # Client SSE path stays unchanged: telemetry + source_coverage still flow to UI.
    sse = captured["sse_tool_result"]
    assert sse["name"] == "find_passages"
    sse_result = sse["result"]
    # _chunks is dropped by strip_internal; the rest (telemetry, source_coverage) survives.
    assert "_chunks" not in sse_result
    assert sse_result["source_coverage"] == _NOISY_FIND_PASSAGES_RESULT["source_coverage"]
    assert sse_result["candidates_evaluated"] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_llm_tool_message_error_path_also_feeds_chunks_only(mock_stream_llm, protocol):
    """Resolution-error path: result dict is `{"_chunks": error, ...}` — same rule applies."""
    error_result = {
        "_chunks": "source 's999' not found.",
        "source_id": None,
        "source_title": "",
        "tool_name": "read_source",
    }
    captured = await _drive_stream_with_tools(mock_stream_llm, protocol=protocol, tool_result=error_result)

    second = captured["second_call_messages"]
    content = _last_tool_result_content(second, protocol)
    assert content == "source 's999' not found."


@pytest.mark.asyncio
async def test_tool_block_sink_still_receives_raw_chunks(mock_stream_llm):
    """`_raw_chunks` must stay in the result dict (tool_block_sink read at chat.py:402)."""
    sink: list[dict] = []
    await _drive_stream_with_tools(
        mock_stream_llm, protocol="openai", tool_result=_NOISY_FIND_PASSAGES_RESULT, tool_block_sink=sink
    )

    assert len(sink) == 1
    block = sink[0]
    # raw_chunks is read by chat.py:402 for tool_block_sink persistence; the dict must keep it.
    # For retrieve-family (find_passages), build_tool_block_entry nests it as result["chunks"].
    assert block["result"]["chunks"] == _NOISY_FIND_PASSAGES_RESULT["_raw_chunks"]


@pytest.mark.asyncio
async def test_end_of_turn_dump_captures_cumulative_state_after_tool_use(
    monkeypatch, tmp_bibilab_home, mock_stream_llm
):
    """T3 bug fix: the end-of-turn dump must contain the cumulative LLM state
    (user + assistant tool_call + tool result), not just the pre-tool state.

    Previously stream_with_tools rebound messages to a defensive local copy, so
    in-loop appends never propagated back to the caller's list — and the dump
    captured only the pre-tool state.
    """
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers import chat as chat_module

    retrieve_tc = ToolCall(id="toolu_t3", name=FIND_PASSAGES_TOOL.name, arguments={"query": "x"})
    call_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=retrieve_tc)
        else:
            yield StreamEvent(type="delta", content="Answer [1]")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute(tool_name, arguments, **kwargs):
        return {
            "query": "x",
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 1,
            "source_coverage": [{"source_id": "s1", "title": "V1"}],
            "_chunks": '[1] "verbatim"',
            "_raw_chunks": [
                {
                    "source_id": "s1",
                    "chunk_id": "v1_0_10",
                    "content": "verbatim",
                    "video_title": "V1",
                    "timestamp_start": 0.0,
                    "timestamp_end": 10.0,
                    "citation_index": 1,
                },
            ],
        }

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "execute_tool", fake_execute)
    monkeypatch.setattr(chat_module, "update_turn_terminal", noop)
    monkeypatch.setattr(chat_module, "maybe_compress_conversation", noop)

    registry = ChatRunRegistry()
    msg_id = "msg-t3-bug"
    registry.register(msg_id, task=None)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(debug_prompts=True),
    )

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="q",
        history=[],
        summary=None,
        source_ids=["s1"],
        ui_lang="en",
        cfg=cfg,
        registry=registry,
        user_msg_id="um-1",
    )

    debug_path = tmp_bibilab_home / "debug" / f"{msg_id}.json"
    assert debug_path.exists(), f"expected dump file at {debug_path}"
    payload = json.loads(debug_path.read_text())

    roles = [m.get("role") for m in payload["messages"]]
    # The bug: the dump had only [user]. The fix: cumulative state includes the
    # tool exchange — assistant(tool_calls) + tool result.
    assert "user" in roles
    assert "tool" in roles, f"dump missing tool role — cumulative state did not propagate. Got: {roles}"
    # Sanity: the user message is first; the cumulative state ends with the
    # next-round narrate-before-acting trigger, appended onto the tool result.
    assert roles[0] == "user"
    from bibilab.routers.chat import _PREAMBLE_TRIGGER

    assert payload["messages"][-1] == {"role": "user", "content": _PREAMBLE_TRIGGER}
    # The assistant message has tool_calls (openai protocol).
    assistant_msgs = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert "tool_calls" in assistant_msgs[0]
    # The tool message is keyed by the tool call id we sent.
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "toolu_t3"


@pytest.mark.asyncio
async def test_messages_sink_export_via_stream_with_tools_directly(mock_stream_llm):
    """Unit-level guard for the messages_sink contract: stream_with_tools
    populates the sink with the cumulative state on every exit path."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.routers import chat as chat_module

    call_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=ToolCall(id="u1", name=TOOL_NAME_FIND_PASSAGES, arguments={}))
        else:
            yield StreamEvent(type="delta", content="x")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute(*args, **kwargs):
        return {"_chunks": "y"}

    sink: list[dict] = []
    gen = chat_module.stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        tools=[],
        execute_tool_fn=fake_execute,
        messages_sink=sink,
    )
    async for _ in gen:
        pass

    roles = [m.get("role") for m in sink]
    assert "tool" in roles
    # Cumulative state: user question + assistant(tool_calls) + tool result, with
    # the next-round narrate-before-acting trigger appended on the tail.
    from bibilab.routers.chat import _PREAMBLE_TRIGGER

    assert sink[-1] == {"role": "user", "content": _PREAMBLE_TRIGGER}


@pytest.mark.asyncio
async def test_messages_sink_populated_on_tool_execution_error(mock_stream_llm):
    """Regression guard: when execute_tool_fn raises, stream_with_tools
    yields SSE_EVENT_ERROR and returns early (chat.py:395). The try/finally
    must still fire and populate the sink with whatever the LLM had seen
    at the point of failure — just the user message, since the failed
    tool call was never appended (the code bails before messages.append)."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.routers import chat as chat_module

    async def fake_stream_llm(messages, cfg, tools=None, system=None):
        yield StreamEvent(type="tool_call", tool_call=ToolCall(id="u1", name=TOOL_NAME_FIND_PASSAGES, arguments={}))

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute(*args, **kwargs):
        raise RuntimeError("boom")

    sink: list[dict] = []
    gen = chat_module.stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        tools=[],
        execute_tool_fn=fake_execute,
        messages_sink=sink,
    )
    events = [e async for e in gen]
    # The error event was yielded.
    assert any(getattr(e, "type", None) == "error" for e in events)

    # The try/finally must fire on this exit path too — sink is populated
    # with the cumulative LLM state at the point of early return. The
    # exact content is implementation-defined (the failed tool call is
    # NOT appended because the code bails before messages.append), so we
    # only assert that the user message the LLM actually saw is present.
    assert len(sink) >= 1
    assert sink[0]["role"] == "user"
    # The preamble trigger is folded into the question for OpenAI, so the user
    # message the LLM saw starts with the question text.
    assert sink[0]["content"].startswith("q")
