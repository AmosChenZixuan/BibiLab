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
async def test_stream_with_tools_records_escalation_iteration_snapshot():
    """Iter-1 find_passages → iter-2 read_source → iter-3 answer.

    Each iteration's snapshot must reflect the messages accumulated so far,
    so the escalation (locator → reader) is auditable from the dump.
    """
    from bibilab.config import AIConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL, READ_SOURCE_TOOL
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="x", api_key="k", base_url="")
    find_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"})
    read_tc = ToolCall(id="c2", name=READ_SOURCE_TOOL.name, arguments={"source_id": "s1"})

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=find_tc)
        elif call_count == 2:
            yield StreamEvent(type="tool_call", tool_call=read_tc)
        else:
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

    async def fake_execute(name, args, **kwargs):
        if name == FIND_PASSAGES_TOOL.name:
            return {
                "ok": True,
                "query": "q",
                "source_coverage": [{"source_id": "s1"}],
                "candidates_evaluated": 1,
                "sources_with_hits": 1,
                "sources_total": 1,
            }
        if name == READ_SOURCE_TOOL.name:
            return {"ok": True, "source_id": "s1", "content": "verbatim"}
        raise ValueError(f"unexpected tool: {name}")

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

    # iter-1 (find_passages) + iter-2 (read_source) + iter-3 (synth) = 3 snapshots
    assert len(sink) == 3

    # iter-1: both tools available; LLM chose find_passages.
    assert [t["name"] for t in sink[0]["tools"]] == [FIND_PASSAGES_TOOL.name, READ_SOURCE_TOOL.name]
    assert sink[0]["messages"] == [{"role": "user", "content": "q"}]

    # iter-2: both tools still available (iter <= MAX); LLM chose read_source.
    assert [t["name"] for t in sink[1]["tools"]] == [FIND_PASSAGES_TOOL.name, READ_SOURCE_TOOL.name]
    # iter-2 messages include the assistant tool_use + tool_result from iter-1's find_passages.
    iter2_msgs = sink[1]["messages"]
    iter2_assistant = [m for m in iter2_msgs if m["role"] == "assistant"]
    assert iter2_assistant, "iter-2 must include the iter-1 assistant tool_use turn"
    iter1_tool_call_ids = [tc["id"] for tc in iter2_assistant[-1]["tool_calls"]]
    assert "c1" in iter1_tool_call_ids
    iter2_tool_msgs = [m for m in iter2_msgs if m["role"] == "tool"]
    assert any(m["tool_call_id"] == "c1" for m in iter2_tool_msgs), "iter-2 must replay iter-1's tool_result"

    # iter-3: synthesis turn — both tools still listed (iter == MAX, not yet > MAX).
    assert [t["name"] for t in sink[2]["tools"]] == [FIND_PASSAGES_TOOL.name, READ_SOURCE_TOOL.name]
    # iter-3 messages include the tool_result from iter-2's read_source (id=c2).
    iter3_tool_msgs = [m for m in sink[2]["messages"] if m["role"] == "tool"]
    assert any(m["tool_call_id"] == "c2" for m in iter3_tool_msgs), "iter-3 must replay iter-2's read_source result"


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


@pytest.mark.asyncio
async def test_run_chat_turn_dumps_trace_when_flag_enabled(monkeypatch, tmp_path: Path):
    """With rag.debug_prompts=True, a JSON file appears at ~/.bibilab/debug/{message_id}.json."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    captured_systems: list = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        captured_systems.append(system)
        yield StreamEvent(type="delta", content="Answer")
        yield StreamEvent(type="done")

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(debug_prompts=True),
    )

    registry = ChatRunRegistry()
    msg_id = "msg-debug-on"
    registry.register(msg_id, task=None)

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="q",
        history=[],
        summary="User previously asked about X",
        source_ids=[],
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )

    dump_path = tmp_path / "debug" / f"{msg_id}.json"
    assert dump_path.exists()
    data = json.loads(dump_path.read_text())
    # system captured from the actual grounding prompt (non-empty)
    assert data["system"] and "Respond in" in data["system"]
    # summary text round-trips through the dumped system prompt
    assert "User previously asked about X" in data["system"]
    # one iteration (no tool calls, direct answer)
    assert len(data["iterations"]) == 1
    # tools list captures both available tools for the iteration
    assert [t["name"] for t in data["iterations"][0]["tools"]] == ["find_passages", "read_source"]
    # iteration captures the user message
    assert data["iterations"][0]["messages"][-1] == {"role": "user", "content": "q"}


@pytest.mark.asyncio
async def test_run_chat_turn_no_dump_when_flag_disabled(monkeypatch, tmp_path: Path):
    """With rag.debug_prompts=False (default), NO file is written AND
    debug_trace_sink is None at the stream_with_tools call site.

    Capturing the kwarg locks in the gating: a regression that always passes
    the sink (e.g. `debug_trace_sink=debug_trace`) would not be caught by the
    file-existence assertion alone, because the sink is silently accumulated
    and never flushed when iterations is empty.
    """
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    captured_kwargs: list = []

    async def fake_stream_with_tools(*args, **kwargs):
        captured_kwargs.append(kwargs)
        yield StreamEvent(type="delta", content="A")
        yield StreamEvent(type="done")

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "stream_with_tools", fake_stream_with_tools)
    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(),  # defaults: debug_prompts=False
    )

    registry = ChatRunRegistry()
    msg_id = "msg-debug-off"
    registry.register(msg_id, task=None)

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="q",
        history=[],
        summary=None,
        source_ids=[],
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )

    debug_dir = tmp_path / "debug"
    assert not (debug_dir / f"{msg_id}.json").exists()

    assert captured_kwargs, "stream_with_tools was not called"
    # When the flag is off, the sink MUST be None — not an empty list.
    # An empty-list default would silently accumulate per-iteration snapshots
    # and only the file-skip in _dump_prompt_trace would hide the leak.
    assert captured_kwargs[0]["debug_trace_sink"] is None


@pytest.mark.asyncio
async def test_run_chat_turn_dump_failure_does_not_break_turn(monkeypatch, tmp_path: Path, caplog):
    """A dump write failure (e.g. permission denied) must not propagate."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig, RagConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="delta", content="A")
        yield StreamEvent(type="done")

    update_calls: list = []

    async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
        update_calls.append((message_id, content, status))

    async def noop_set(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
    monkeypatch.setattr(chat_module, "update_message_content", capture_update)
    monkeypatch.setattr(chat_module, "set_active_stream", noop_set)

    # bibilab_home points to a path with a blocking file at the debug/ location
    home = tmp_path
    (home / "debug").write_text("blocking file")

    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: home)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
        rag=RagConfig(debug_prompts=True),
    )

    registry = ChatRunRegistry()
    msg_id = "msg-debug-fail"
    registry.register(msg_id, task=None)

    with caplog.at_level(logging.WARNING, logger="bibilab.routers.chat"):
        await chat_module.run_chat_turn(
            message_id=msg_id,
            conversation_id="c1",
            list_id="l1",
            user_message_text="q",
            history=[],
            summary=None,
            source_ids=[],
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )

    # Turn succeeded: assistant message persisted with status='done'
    assert any(call[0] == msg_id and call[2] == "done" for call in update_calls)
    # Dump failure was logged
    assert any("dump_prompt_trace_failed" in r.message for r in caplog.records)
