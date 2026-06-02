"""Tests for opt-in per-message LLM prompt-trace dump (#393)."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from bibilab.config import RagConfig


def test_rag_config_default_debug_prompts_is_false():
    """Off by default — opt-in flag, zero behavior change for existing users."""
    cfg = RagConfig()
    assert cfg.debug_prompts is False


def test_dump_prompt_trace_writes_json(monkeypatch, tmp_path: Path):
    """Helper writes a valid JSON file with system + iterations."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    iterations = [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "find_passages", "description": "x", "parameters": {}}],
        }
    ]
    _dump_prompt_trace("m1", "sys prompt", iterations, debug_dir)

    out = debug_dir / "m1.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data == {
        "system": "sys prompt",
        "iterations": iterations,
    }


def test_dump_prompt_trace_skips_empty_iterations(tmp_path: Path):
    """If there are no iterations, do not write a file (nothing to debug)."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    _dump_prompt_trace("m1", "sys", [], debug_dir)

    assert not (debug_dir / "m1.json").exists()
    assert not debug_dir.exists()


def test_dump_prompt_trace_allows_null_system(tmp_path: Path):
    """system may be None (e.g. whitespace-only after build_grounding_prompt) — store as null."""
    from bibilab.routers.chat import _dump_prompt_trace

    debug_dir = tmp_path / "debug"
    _dump_prompt_trace("m1", None, [{"messages": [], "tools": []}], debug_dir)

    data = json.loads((debug_dir / "m1.json").read_text())
    assert data["system"] is None


def test_dump_prompt_trace_swallows_write_errors(tmp_path: Path, caplog):
    """A write failure (e.g. permission denied) must not propagate — it must log."""
    from bibilab.routers import chat as chat_module

    # Use a path where mkdir will fail: a file exists at the location.
    bad = tmp_path / "blocking_file"
    bad.write_text("not a dir")
    # _dump_prompt_trace calls debug_dir.mkdir(parents=True, exist_ok=True)
    # which will raise NotADirectoryError on Linux when parent is a file.

    with caplog.at_level(logging.WARNING, logger="bibilab.routers.chat"):
        chat_module._dump_prompt_trace(
            "m1",
            "sys",
            [{"messages": [], "tools": []}],
            bad,
        )
    # No exception propagated. The blocking file is untouched.
    assert bad.read_text() == "not a dir"
    assert any("dump_prompt_trace_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stream_with_tools_records_per_iteration_snapshot():
    """Each stream_llm call (iter-1, iter-2, forced synth) records a snapshot
    in the debug_trace_sink with the messages+tools that will be sent."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL, READ_SOURCE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        return {
            "ok": True,
            "query": "q",
            "source_coverage": [],
            "candidates_evaluated": 1,
            "sources_with_hits": 0,
            "sources_total": 1,
            "source_id": "s1",
        }

    sink: list = []
    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[FIND_PASSAGES_TOOL, READ_SOURCE_TOOL],
            execute_tool_fn=fake_execute,
            debug_trace_sink=sink,
        ):
            pass

    # iter-1 (find_passages) + iter-2 (synthesis with both tools) = 2 snapshots
    assert len(sink) == 2
    # iter-1: both tools available
    assert [t["name"] for t in sink[0]["tools"]] == [FIND_PASSAGES_TOOL.name, READ_SOURCE_TOOL.name]
    assert sink[0]["messages"] == [{"role": "user", "content": "q"}]
    # iter-2: still both tools (synthesis not yet triggered; iter <= MAX)
    assert [t["name"] for t in sink[1]["tools"]] == [FIND_PASSAGES_TOOL.name, READ_SOURCE_TOOL.name]
    # iter-2 messages include the tool_exchange appended after iter-1
    assert any(m["role"] == "tool" for m in sink[1]["messages"])


@pytest.mark.asyncio
async def test_stream_with_tools_synthesis_turn_records_empty_tools():
    """The synthesis turn (iter > MAX_TOOL_ITERATIONS) is recorded with tools=[]."""
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers.chat import MAX_TOOL_ITERATIONS, stream_with_tools

    cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        else:
            yield StreamEvent(type="done")  # empty synthesis → triggers forced synth

    async def fake_execute(name, args, **kwargs):
        return {
            "ok": True,
            "query": "q",
            "source_coverage": [],
            "candidates_evaluated": 1,
            "sources_with_hits": 0,
            "sources_total": 1,
        }

    sink: list = []
    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream):
        async for _ in stream_with_tools(
            messages=[{"role": "user", "content": "q"}],
            cfg=cfg,
            tools=[FIND_PASSAGES_TOOL],
            execute_tool_fn=fake_execute,
            debug_trace_sink=sink,
        ):
            pass

    # MAX_TOOL_ITERATIONS tool turns + 1 empty synthesis + 1 forced synth = 5 snapshots
    assert len(sink) == MAX_TOOL_ITERATIONS + 2
    # Synthesis (iter MAX+1) and forced synth both have no tools
    assert sink[MAX_TOOL_ITERATIONS]["tools"] == []
    assert sink[MAX_TOOL_ITERATIONS + 1]["tools"] == []
