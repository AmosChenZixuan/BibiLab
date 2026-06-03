"""Tests for opt-in per-LLM-call dump (input + response) (#399)."""

import json
import logging
from pathlib import Path

import pytest

from bibilab.pipeline._shared import StreamEvent, ToolDefinition
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
async def test_stream_with_tools_dumps_one_file_per_llm_call(monkeypatch, tmp_path: Path):
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

    monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)

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
