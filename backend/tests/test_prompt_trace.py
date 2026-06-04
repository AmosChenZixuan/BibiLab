"""Tests for opt-in per-LLM-call dump (input + response) (#399) and the
LLM-bound tool message content (#401 — feed `_chunks` only, no FTS noise)."""

import json
import logging
from pathlib import Path

import pytest

from bibilab.config import AIConfig
from bibilab.pipeline._shared import StreamEvent, ToolCall, ToolDefinition
from bibilab.routers.chat import _dump_turn


def test_dump_turn_writes_input_and_response(tmp_path: Path):
    """Core: one file captures system, tools, messages, and the LLM response."""
    tools = [
        ToolDefinition(
            name="find_passages",
            description="locator",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    _dump_turn(
        debug_dir=tmp_path,
        llm_call=1,
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=tools,
        response_text="answer",
        response_tool_calls=[{"id": "c1", "name": "find_passages", "arguments": {"query": "q"}}],
    )
    payload = json.loads((tmp_path / "call1.json").read_text())
    assert payload == {
        "system": "sys",
        "tools": [{"name": "find_passages", "description": "locator", "parameters": tools[0].parameters}],
        "messages": [{"role": "user", "content": "q"}],
        "response": {
            "text": "answer",
            "tool_calls": [{"id": "c1", "name": "find_passages", "arguments": {"query": "q"}}],
        },
    }


def test_dump_turn_preserves_cjk(tmp_path: Path):
    """CJK must never be escaped to \\uXXXX anywhere in the dump — including
    pre-serialized tool-call args and tool_result content."""
    args_str = json.dumps({"query": "演讲中提到的主要观点"}, ensure_ascii=False)
    _dump_turn(
        debug_dir=tmp_path,
        llm_call=1,
        system="Respond in 简体中文. Use 第八集 for episodes.",
        messages=[
            {"role": "user", "content": "第三集讲了什么？"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "find_passages", "arguments": args_str}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": json.dumps({"title": "第三集 — 高潮"}, ensure_ascii=False),
                    }
                ],
            },
        ],
        tools=[],
        response_text="我来回答…",
    )
    text = (tmp_path / "call1.json").read_text()
    assert "\\u" not in text
    for s in ("第八集", "第三集讲了什么？", "演讲中提到的主要观点", "第三集 — 高潮", "我来回答"):
        assert s in text


def test_dump_turn_swallows_errors(tmp_path: Path, caplog):
    """A write failure must be logged, not propagated."""
    from bibilab.routers import chat as chat_module

    blocking = tmp_path / "blocker"
    blocking.write_text("not a dir")  # debug_dir points at a file -> write fails
    with caplog.at_level(logging.WARNING, logger="bibilab.routers.chat"):
        chat_module._dump_turn(debug_dir=blocking, llm_call=1, system="sys", messages=[], tools=[])
    assert blocking.read_text() == "not a dir"
    assert any("dump_turn_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stream_with_tools_dumps_one_file_per_llm_call(monkeypatch, tmp_path: Path, mock_stream_llm):
    """Integration: each LLM call gets its own file; call1 captures the tool_call response."""
    from bibilab.config import AIConfig
    from bibilab.routers import chat as chat_module

    calls = {"n": 0}

    async def fake_stream_llm(messages, cfg, tools, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = type("T", (), {"id": "toolu_a", "name": "find_passages", "arguments": {"query": "q"}})()
            yield StreamEvent(type="tool_call", tool_call=tc)
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        return {"_chunks": "excerpt"}

    mock_stream_llm.side_effect = fake_stream_llm

    cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="")
    gen = chat_module.stream_with_tools(
        messages=[{"role": "user", "content": "q"}],
        cfg=cfg,
        tools=[],
        execute_tool_fn=fake_execute,
        debug_dump_dir=tmp_path,
    )
    async for _ in gen:
        pass

    assert (tmp_path / "call2.json").exists()
    call1 = json.loads((tmp_path / "call1.json").read_text())
    assert call1["response"]["tool_calls"] == [{"id": "toolu_a", "name": "find_passages", "arguments": {"query": "q"}}]


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
    "tool_name": "find_passages",
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
            tc = ToolCall(id="toolu_a", name="find_passages", arguments={"query": "q"})
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


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_llm_tool_message_feeds_only_chunks_excerpts_no_noise(mock_stream_llm, protocol):
    """#401: LLM tool message content is `_chunks` only — not the full result dict.
    Asserts the symmetric Anthropic + OpenAI branches; client SSE payload is unchanged.
    """
    captured = await _drive_stream_with_tools(
        mock_stream_llm, protocol=protocol, tool_result=_NOISY_FIND_PASSAGES_RESULT
    )

    second = captured["second_call_messages"]
    # The tool message is appended after the original user message; take the last one
    # with role in (user, tool) — that's the loopback tool message.
    tool_messages = [m for m in second if m.get("role") in ("user", "tool")]
    assert tool_messages, f"expected a tool message, got {second}"
    msg = tool_messages[-1]
    if protocol == "anthropic":
        # Anthropic: tool_result is the sole content block.
        assert msg["role"] == "user"
        blocks = msg["content"]
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "tool_result"
        content = block["content"]
    else:
        # OpenAI: tool message with content string.
        assert msg["role"] == "tool"
        content = msg["content"]

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
    tool_messages = [m for m in second if m.get("role") in ("user", "tool")]
    msg = tool_messages[-1]
    if protocol == "anthropic":
        content = msg["content"][0]["content"]
    else:
        content = msg["content"]
    assert content == "source 's999' not found."


@pytest.mark.asyncio
async def test_tool_block_sink_still_receives_raw_chunks(mock_stream_llm):
    """#401 constraint: `_raw_chunks` must stay in the result dict (tool_block_sink read at chat.py:402)."""
    sink: list[dict] = []
    await _drive_stream_with_tools(
        mock_stream_llm, protocol="openai", tool_result=_NOISY_FIND_PASSAGES_RESULT, tool_block_sink=sink
    )

    assert len(sink) == 1
    block = sink[0]
    # raw_chunks is read by chat.py:402 for tool_block_sink persistence; the dict must keep it.
    # For retrieve-family (find_passages), build_tool_block_entry nests it as result["chunks"].
    assert block["result"]["chunks"] == _NOISY_FIND_PASSAGES_RESULT["_raw_chunks"]
