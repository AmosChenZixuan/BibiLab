"""Tests for the SSE chat streaming endpoint (post-tool-calling refactor)."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from bibilab.pipeline._shared import StreamEvent, ToolCall
from bibilab.routers.chat import (
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_TOOL_RESULT,
    _client_tool_result,
)
from tests import an_async_generator
from tests.factories import MessageFactory

pytestmark = pytest.mark.integration


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


async def _get_assistant_msgs(client, list_id: str) -> list[dict]:
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    return [m for m in conv_resp.json()["messages"] if m["role"] == "assistant"]


async def _create_list(client, name: str) -> str:
    return (await client.post("/lists", json={"name": name})).json()["id"]


@pytest.mark.asyncio
async def test_chat_endpoint_returns_sse_stream(client):
    """POST /lists/:id/chat returns text/event-stream with delta events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator(
            [
                StreamEvent(type="delta", content="Hello"),
                StreamEvent(type="delta", content=" world"),
                StreamEvent(type="done"),
            ]
        )
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_endpoint_404_for_missing_list(client):
    """Returns 404 when list does not exist."""
    resp = await client.post("/lists/nonexistent/chat", json={"message": "hi"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_endpoint_saves_user_message(client):
    """User message is persisted to DB after successful stream."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator(
            [
                StreamEvent(type="delta", content="Hi"),
                StreamEvent(type="done"),
            ]
        )
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hello"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    messages = conv_resp.json()["messages"]
    assert any(m["role"] == "user" and m["content"] == "hello" for m in messages)


@pytest.mark.asyncio
async def test_chat_endpoint_saves_assistant_message(client):
    """Assistant message is persisted to DB after stream completes."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator(
            [
                StreamEvent(type="delta", content="Hello"),
                StreamEvent(type="done"),
            ]
        )
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    messages = conv_resp.json()["messages"]
    assert any(m["role"] == "assistant" and m["content"] == "Hello" for m in messages)


@pytest.mark.asyncio
async def test_chat_endpoint_uses_conversation_history(client):
    """Prior conversation messages are included in LLM context."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    from bibilab.db import get_or_create_conversation

    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(
        conv_id,
        content="Previous question",
    )
    await MessageFactory.build(
        conv_id,
        role="assistant",
        content="Previous answer",
    )

    captured_messages = []

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        captured_messages.append(messages)
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", capture):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "follow-up"})

    assert resp.status_code == 200
    roles = [m["role"] for m in captured_messages[0]]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_chat_endpoint_includes_tools(client):
    """find_passages and read_source tools are passed to stream_with_tools."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL, READ_SOURCE_TOOL

    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    captured_tools = []

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        captured_tools.append(tools)
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", capture):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    tool_names = [t.name for t in captured_tools[0]]
    assert FIND_PASSAGES_TOOL.name in tool_names
    assert READ_SOURCE_TOOL.name in tool_names


@pytest.mark.asyncio
async def test_retrieve_tool_result_has_coverage(client):
    """find_passages tool_result includes coverage metadata."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        result = await execute_tool_fn(FIND_PASSAGES_TOOL.name, {"query": "test"})
        yield StreamEvent(
            type="tool_result",
            content=json.dumps({"name": FIND_PASSAGES_TOOL.name, "result": _client_tool_result(result)}),
        )
        yield StreamEvent(type="delta", content="Answer")
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "what is this?"})

    assert resp.status_code == 200
    assert "tool_result" in resp.text
    assert "sources_total" in resp.text


@pytest.mark.asyncio
async def test_conversation_no_longer_has_mode(client):
    """After mode column dropped, conversation response has no mode field."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator([StreamEvent(type="done")])
        await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    assert "mode" not in conv_resp.json()["conversation"]


@pytest.mark.asyncio
async def test_patch_conversation_endpoint_gone(client):
    """PATCH /lists/:id/conversation returns 405 (no route)."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.patch(f"/lists/{list_id}/conversation", json={"mode": "broad"})
    assert resp.status_code == 405


# ---------------------------------------------------------------------------
# Smoke tests: 4 scenarios verifying LLM tool-calling behaviour end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_scenario_1_chitchat_no_tool_calls(client):
    """Chitchat: LLM responds directly, no retrieve tool, no tool_result in SSE."""
    list_id = await _create_list(client, "Smoke 1")

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(type=SSE_EVENT_DELTA, content="You're welcome!")
        yield StreamEvent(type=SSE_EVENT_DONE)

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "thanks"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert SSE_EVENT_TOOL_RESULT not in types, "chitchat should NOT trigger any tool"
    assert SSE_EVENT_DELTA in types
    assert SSE_EVENT_DONE in types

    assistant_msgs = await _get_assistant_msgs(client, list_id)
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "You're welcome!"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,query,user_message,delta_text,result_overrides",
    [
        (
            "find_passages",
            "test query",
            "What does the video say about transformers?",
            "Based on the transcript, the answer is...",
            {
                "candidates_evaluated": 30,
                "sources_with_hits": 1,
                "sources_total": 1,
                "source_coverage": [{"source_id": "src-1", "title": "Test Video"}],
                "_chunks": [{"title": "Test Video", "start": 10.0, "end": 25.0, "content": "relevant chunk"}],
            },
        ),
    ],
)
async def test_smoke_scenario_2_retrieve(client, tool_name, query, user_message, delta_text, result_overrides):
    """LLM calls find_passages tool with correct tool_name before answering."""
    list_id = await _create_list(client, f"Smoke retrieve {tool_name}")

    retrieve_result = {"tool_name": tool_name, **result_overrides}

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name=tool_name, arguments={"query": query}),
        )
        result = await execute_tool_fn(tool_name, {"query": query})
        tool_result_data = {"name": tool_name, "result": _client_tool_result(result)}
        yield StreamEvent(type="tool_result", content=json.dumps(tool_result_data))
        yield StreamEvent(type=SSE_EVENT_DELTA, content=delta_text)
        yield StreamEvent(type=SSE_EVENT_DONE)

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = retrieve_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": user_message})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]

    assert SSE_EVENT_TOOL_RESULT in types
    tool_results = [e for e in events if e["type"] == SSE_EVENT_TOOL_RESULT]
    assert len(tool_results) == 1, "one tool_result from stream (metadata saved to DB, not SSE)"

    stream_result = tool_results[0]
    assert stream_result["name"] == tool_name
    assert stream_result["result"]["candidates_evaluated"] == result_overrides["candidates_evaluated"]
    assert "_chunks" not in stream_result["result"], "_chunks must be stripped before sending to client"

    assert SSE_EVENT_DELTA in types
    assert SSE_EVENT_DONE in types

    assistant_msgs = await _get_assistant_msgs(client, list_id)
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["metadata"] is not None
    assert assistant_msgs[0]["metadata"]["rag"]["calls"][0]["tool_name"] == tool_name


@pytest.mark.asyncio
async def test_chat_sse_emits_citation_events(client):
    """SSE stream includes citation events interleaved with deltas, no [N] in text."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream(*args, **kwargs):
        async for ev in an_async_generator(
            [
                StreamEvent(type="delta", content="See "),
                StreamEvent(type="citation", content='{"index":1,"source_id":"s1"}'),
                StreamEvent(type="delta", content=" for details."),
                StreamEvent(type="done"),
            ]
        ):
            yield ev

    with patch("bibilab.routers.chat.stream_with_tools", side_effect=fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "citation" in types
    assert "done" in types
    delta_text = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "[1]" not in delta_text


@pytest.mark.asyncio
async def test_chat_persists_content_blocks_in_metadata(client):
    """Assistant message metadata includes content_blocks after citation stream."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream(*args, **kwargs):
        async for ev in an_async_generator(
            [
                StreamEvent(type="delta", content="Hello "),
                StreamEvent(type="citation", content='{"index":1,"source_id":"s1"}'),
                StreamEvent(type="delta", content=" world"),
                StreamEvent(type="done"),
            ]
        ):
            yield ev

    with patch("bibilab.routers.chat.stream_with_tools", side_effect=fake_stream):
        await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    msgs = await _get_assistant_msgs(client, list_id)
    assert len(msgs) == 1
    meta = msgs[0].get("metadata")
    assert meta is not None
    blocks = meta["content_blocks"]
    assert len(blocks) == 3
    assert blocks[0] == {"type": "text", "text": "Hello "}
    assert blocks[1]["type"] == "citation" and blocks[1]["index"] == 1
    assert blocks[2] == {"type": "text", "text": " world"}


@pytest.mark.asyncio
async def test_chat_persists_paragraph_breaks_in_content_blocks(client):
    """Paragraph boundaries (\n\n+) are split into paragraph_break blocks server-side.

    Exercises both flush sites: mid-stream citation flush (pending text before
    a citation) and post-stream final flush.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream(*args, **kwargs):
        async for ev in an_async_generator(
            [
                StreamEvent(type="delta", content="First paragraph "),
                StreamEvent(type="citation", content='{"index":1,"source_id":"s1"}'),
                StreamEvent(type="delta", content=".\n\nSecond paragraph "),
                StreamEvent(type="citation", content='{"index":2,"source_id":"s2"}'),
                StreamEvent(type="delta", content="."),
                StreamEvent(type="done"),
            ]
        ):
            yield ev

    with patch("bibilab.routers.chat.stream_with_tools", side_effect=fake_stream):
        await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    msgs = await _get_assistant_msgs(client, list_id)
    blocks = msgs[0]["metadata"]["content_blocks"]
    types = [b["type"] for b in blocks]
    # Expected: text, citation, text, paragraph_break, text, citation, text
    assert types == ["text", "citation", "text", "paragraph_break", "text", "citation", "text"]
    assert blocks[2]["text"] == "."
    assert blocks[4]["text"] == "Second paragraph "


@pytest.mark.asyncio
async def test_chat_citation_after_lone_break_not_isolated(client):
    """`a\\n\\n[1]` persists as [text, paragraph_break, citation].

    Backend no longer pops the paragraph_break before a citation — the frontend
    post-merge fold in renderParagraphs handles citation-only trailing paragraphs.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream(*args, **kwargs):
        async for ev in an_async_generator(
            [
                StreamEvent(type="delta", content="a\n\n"),
                StreamEvent(type="citation", content='{"index":1,"source_id":"s1"}'),
                StreamEvent(type="done"),
            ]
        ):
            yield ev

    with patch("bibilab.routers.chat.stream_with_tools", side_effect=fake_stream):
        await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    msgs = await _get_assistant_msgs(client, list_id)
    blocks = msgs[0]["metadata"]["content_blocks"]
    assert len(blocks) == 3
    assert blocks[0] == {"type": "text", "text": "a"}
    assert blocks[1] == {"type": "paragraph_break"}
    assert blocks[2] == {"type": "citation", "index": 1, "source_id": "s1", "chunk_ids": []}


@pytest.mark.asyncio
async def test_chat_sse_multi_find_passages_no_crash(client):
    """Smoke test: two find_passages calls in one turn do not crash the SSE stream.

    The registry ordering and dedup logic is exercised in
    test_chat_tools.py; this test verifies the SSE layer handles the
    multi-call flow without internal errors.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_with_tools(*args, **kwargs):
        tc1 = ToolCall(id="tc1", name="find_passages", arguments={"query": "q1"})
        tc2 = ToolCall(id="tc2", name="find_passages", arguments={"query": "q2"})
        yield StreamEvent(type="tool_call", tool_call=tc1)
        yield StreamEvent(type="done")
        yield StreamEvent(type="tool_call", tool_call=tc2)
        yield StreamEvent(type="done")
        yield StreamEvent(type="delta", content="result")
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", side_effect=fake_stream_with_tools):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "done" in types


@pytest.mark.asyncio
async def test_pure_ack_no_retrieve_called(client, mock_stream_llm):
    """
    User sends '嗯' — LLM yields text directly, no tool_call. trivial_ack_note
    is deleted; this verifies behavior relies on LLM tool_choice.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    tool_calls_seen: list[str] = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        # LLM looks at "嗯", chooses to reply with text only.
        yield StreamEvent(type="delta", content="好的，您还有别的问题吗？")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        tool_calls_seen.append(kwargs.get("tool_name", "?"))
        return {}

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "嗯"})

    assert resp.status_code == 200
    assert b"tool_call_start" not in resp.content
    assert tool_calls_seen == [], f"expected no tool calls, got {tool_calls_seen}"


@pytest.mark.asyncio
async def test_deltas_streamed_immediately_during_tool_iteration(client, mock_stream_llm):
    """Deltas from a tool iteration reach the client immediately (not buffered until stream end)."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(type="delta", content="Let me look that up again.")
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "x"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="final answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 0,
            "sources_with_hits": 0,
            "sources_total": 1,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    body = resp.text
    # Deltas are now streamed immediately, so preamble reaches the client
    assert "Let me look that up again" in body, "preamble from tool iteration must reach client"
    assert "final answer" in body, "terminal iteration's delta must reach the client"


@pytest.mark.asyncio
async def test_terminal_iteration_deltas_streamed(client, mock_stream_llm):
    """A no-tool iteration's deltas must reach the client verbatim."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        yield StreamEvent(type="delta", content="hello world")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm
    resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "hello world" in resp.text


@pytest.mark.asyncio
async def test_chat_sse_hallucinated_index_emitted_as_text(client, caplog, mock_stream_llm):
    """[7] when registry only has 1-2 → emitted as text delta, warning logged.

    Mocks stream_llm (not stream_with_tools) so the real parser runs.
    The registry stays empty (no retrieve triggered), so [7] is out of range
    and the parser logs citation_hallucinated_index.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_llm(*args, **kwargs):
        yield StreamEvent(type="delta", content="see [7]")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    import logging

    with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.citation_parser"):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert all(e["type"] in ("delta", "done", "meta") for e in events)
    delta_text = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "[7]" in delta_text
    assert any("citation_hallucinated_index" in rec.message for rec in caplog.records)


# (audit prior claims rule removed — Rule 3 deferred to #301 fact-check tool)


# ---------------------------------------------------------------------------
# Task 10: tool_blocks persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chat_turn_persists_tool_blocks(monkeypatch, tmp_path, mock_stream_llm):
    """When the producer runs a find_passages tool, tool_blocks must be persisted via update_message_content."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.routers import chat as chat_module

    call_count = 0
    retrieve_tc = ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "x"})

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
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
            "_chunks": "fmt",
            "_turn_indices": [1],
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

    monkeypatch.setattr(chat_module, "execute_tool", fake_execute)

    captured: dict = {}

    async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
        captured["tool_blocks"] = tool_blocks

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "update_message_content", capture_update)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    from bibilab.pipeline.chat_runs import ChatRunRegistry

    registry = ChatRunRegistry()
    msg_id = "msg-1"
    registry.register(msg_id, task=None)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
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
    )

    assert captured.get("tool_blocks") is not None
    assert len(captured["tool_blocks"]) == 1
    assert captured["tool_blocks"][0]["name"] == FIND_PASSAGES_TOOL.name
    assert captured["tool_blocks"][0]["result"]["chunks"][0]["content"] == "verbatim"


@pytest.mark.asyncio
async def test_system_message_is_stable_across_turns(monkeypatch, mock_stream_llm):
    """#310: for a fixed response_language the assembled system prompt is
    byte-identical turn to turn (no per-turn source list) — prompt-cache stable."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    captured_systems: list[str] = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        captured_systems.append(system)
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )

    for _ in range(2):
        registry = ChatRunRegistry()
        msg_id = "msg-stable"
        registry.register(msg_id, task=None)
        await chat_module.run_chat_turn(
            message_id=msg_id,
            conversation_id="c1",
            list_id="l1",
            user_message_text="q",
            history=[],
            summary=None,
            source_ids=["s1", "s2"],
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )

    assert len(captured_systems) == 2
    assert captured_systems[0] == captured_systems[1]
    assert "Sources (scan" not in captured_systems[0]
    assert "Empty list is fine" not in captured_systems[0]


@pytest.mark.asyncio
async def test_tool_call_start_emitted_before_tool_result(client, mock_stream_llm):
    """Each find_passages tool_call must emit a tool_call_start SSE event before its tool_result."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "noodles"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 5,
            "sources_with_hits": 2,
            "sources_total": 3,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    body = resp.text
    start_idx = body.find('"tool_call_start"')
    result_idx = body.find('"tool_result"')
    assert start_idx >= 0, "tool_call_start event missing"
    assert result_idx >= 0, "tool_result event missing"
    assert start_idx < result_idx, "tool_call_start must precede tool_result"
    assert f'"name": "{FIND_PASSAGES_TOOL.name}"' in body
    assert '"query": "noodles"' in body
    assert f'"tool_name": "{FIND_PASSAGES_TOOL.name}"' in body


@pytest.mark.asyncio
async def test_rag_metadata_persists_calls_list(client, mock_stream_llm):
    """metadata.rag.calls must contain one entry per find_passages call in the turn."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "A"}),
            )
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc2", name=FIND_PASSAGES_TOOL.name, arguments={"query": "B"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 5,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})
    assert resp.status_code == 200

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    rag = assistant_msgs[-1]["metadata"]["rag"]
    assert "calls" in rag
    assert len(rag["calls"]) == 2
    queries = sorted(c["query"] for c in rag["calls"])
    assert queries == ["A", "B"]
    assert all(c["tool_name"] == FIND_PASSAGES_TOOL.name for c in rag["calls"])


@pytest.mark.asyncio
async def test_chat_endpoint_error_reason_in_sse_terminal_payload(client):
    """When stream_with_tools yields an error event, the terminal SSE payload
    carries the correct error code message, not the hard-coded default."""
    from unittest.mock import AsyncMock, patch

    list_id = (await client.post("/lists", json={"name": "ErrReason"})).json()["id"]

    class RecordingStreamEvent:
        def __init__(self, type, content=None, tool_call=None):
            self.type = type
            self.content = content
            self.tool_call = tool_call

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield RecordingStreamEvent(type="error", content="original error text from tool failure")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock, return_value={"ok": True}):
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "trigger error"})

    assert resp.status_code == 200
    body = resp.text
    events = []
    for line in body.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1, "terminal error event must be present"
    terminal = error_events[-1]
    # tool_error is the code set when stream_with_tools yields type="error"
    assert terminal["message"] == "tool_error"


@pytest.mark.asyncio
async def test_chat_endpoint_classify_error_reaches_sse_terminal_payload(client, mock_stream_llm):
    """When an SDK exception propagates from stream_llm, classify_error() maps
    it to the correct i18n code in the SSE terminal payload (spec #266 requirement)."""
    list_id = (await client.post("/lists", json={"name": "ClassifyErr"})).json()["id"]

    async def fake_stream_llm_raises(*args, **kwargs):
        raise openai.RateLimitError(
            message="too many requests",
            response=httpx.Response(429, request=httpx.Request("POST", "http://test")),
            body=None,
        )
        yield  # unreachable, makes this an async generator

    mock_stream_llm.side_effect = fake_stream_llm_raises
    resp = await client.post(f"/lists/{list_id}/chat", json={"message": "trigger rate limit"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1, "terminal error event must be present"
    terminal = error_events[-1]
    assert terminal["message"] == "llm_rate_limit_error"


# ---------------------------------------------------------------------------
# Task 5: response_language threading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 12: tool_blocks history expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chat_turn_drops_tool_blocks_on_turn_2(monkeypatch, mock_stream_llm):
    """v2 spec §5.5: prior-turn find_passages / read_source tool exchanges are
    dropped from cross-turn replay to avoid stale-context contamination. The
    LLM sees only the synthesized prose from prior turns; to re-ground on
    prior evidence it must re-call the tool in the current turn.
    """
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.routers import chat as chat_module

    captured_messages: list[list[dict]] = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        captured_messages.append(list(messages))
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    from bibilab.pipeline.chat_runs import ChatRunRegistry

    registry = ChatRunRegistry()
    msg_id = "msg-turn-2"
    registry.register(msg_id, task=None)

    history = [
        {"role": "user", "content": "tell me about quantum mechanics"},
        {
            "role": "assistant",
            "content": "first answer [1]",
            "tool_blocks": [
                {
                    "tool_use_id": "toolu_a",
                    "name": "find_passages",
                    "arguments": {"query": "x"},
                    "result": {"chunks": [{"content": "verbatim"}], "summary": {"sources_total": 1}},
                }
            ],
        },
    ]

    cfg = BibilabConfig(
        ai=AIConfig(protocol="anthropic", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="tell me more about quantum mechanics",
        history=history,
        summary=None,
        source_ids=[],
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )

    assert captured_messages, "stream_llm not called"
    sent = captured_messages[0]
    # v2 drops find_passages blocks; expect user → assistant (text only) → new user
    roles = [m["role"] for m in sent]
    assert roles == ["user", "assistant", "user"]
    # The assistant message carries only the synthesized text — no tool_use expansion
    assert sent[1] == {"role": "assistant", "content": "first answer [1]"}
    # No tool_use / tool_result blocks leaked into any message
    for m in sent:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") not in ("tool_use", "tool_result"), (
                    f"v2 dropped prior tool exchanges but {block.get('type')} leaked into {m['role']}"
                )


@pytest.mark.asyncio
async def test_chat_uses_resolved_response_language_in_system_prompt(monkeypatch, tmp_path, mock_stream_llm):
    """When UI lang is zh and output_language is 'ui', the system prompt must say 'Respond in zh.'."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.routers import chat as chat_module

    captured_system: list[str] = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        captured_system.append(system or "")
        from bibilab.pipeline._shared import StreamEvent

        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    # Build a minimal cfg with output_language="ui" so ui_lang wins
    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url="", output_language="ui"),
        backend=BackendConfig(),
    )

    # Drive run_chat_turn directly with ui_lang="zh"
    from bibilab.pipeline.chat_runs import ChatRunRegistry

    registry = ChatRunRegistry()
    msg_id = "test-msg-lang-1"
    registry.register(msg_id, task=None)

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="conv-1",
        list_id="list-1",
        user_message_text="hi",
        history=[],
        summary=None,
        source_ids=[],
        ui_lang="zh",
        cfg=cfg,
        registry=registry,
    )

    assert captured_system, "stream_llm was never called"
    assert captured_system[0].startswith("Respond in zh."), (
        f"Expected system prompt to start with 'Respond in zh.', got: {captured_system[0][:100]}"
    )


@pytest.mark.asyncio
async def test_run_chat_turn_reseeds_citation_registry_from_history_tool_blocks(monkeypatch):
    """When history contains retrieve tool_blocks, reseed populates citation_registry before stream_with_tools."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.routers import chat as chat_module

    captured_registry: list[dict] = []

    async def fake_stream_with_tools(
        messages,
        cfg,
        tools,
        execute_tool_fn,
        system=None,
        llm_max_tokens=2048,
        registry=None,
        tool_block_sink=None,
        debug_dump_dir=None,
    ):
        # Capture a copy of registry after reseed has populated it.
        captured_registry.append(dict(registry) if registry else {})
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    monkeypatch.setattr(chat_module, "stream_with_tools", fake_stream_with_tools)

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "update_message_content", noop)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    from bibilab.pipeline.chat_runs import ChatRunRegistry

    registry = ChatRunRegistry()
    msg_id = "msg-reseed-1"
    registry.register(msg_id, task=None)

    history = [
        {"role": "user", "content": "tell me about quantum mechanics"},
        {
            "role": "assistant",
            "content": "first answer [1]",
            "tool_blocks": [
                {
                    "tool_use_id": "toolu_a",
                    "name": "find_passages",
                    "arguments": {"query": "x"},
                    "result": {
                        "chunks": [
                            {
                                "source_id": "s1",
                                "citation_index": 1,
                                "chunk_id": "v1_0_10",
                                "video_title": "Video One",
                                "content": "verbatim",
                            },
                            {
                                "source_id": "s1",
                                "citation_index": 1,
                                "chunk_id": "v1_30_40",
                                "video_title": "Video One",
                                "content": "more text",
                            },
                        ],
                        "summary": {"sources_total": 1},
                    },
                }
            ],
        },
    ]

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="tell me more about quantum mechanics",
        history=history,
        summary=None,
        source_ids=["s1"],
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )

    assert captured_registry, "stream_with_tools was not called"
    reg = captured_registry[0]
    assert "s1" in reg, f"Expected registry to contain 's1', got keys: {list(reg.keys())}"
    entry = reg["s1"]
    assert entry.index == 1
    assert entry.source_id == "s1"
    assert entry.title == "Video One"
    assert "v1_0_10" in entry.chunk_ids
    assert "v1_30_40" in entry.chunk_ids


@pytest.mark.asyncio
async def test_rag_call_carries_facet_scope(client, mock_stream_llm):
    """#309: facet_scope from execute_find_passages must survive the router copy into metadata.rag."""
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "FacetTel"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id="tc1",
                    name=FIND_PASSAGES_TOOL.name,
                    arguments={"query": "第八集", "sequence_number": 8},
                ),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 5,
            "source_coverage": [],
            "scoped_pool_size": 5,
            "facet_scope": {
                "sequence_number": 8,
                "season_number": None,
                "matched_count": 2,
                "no_match": False,
            },
            "_chunks": "",
            "_turn_indices": [],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})
    assert resp.status_code == 200

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    rag = assistant_msgs[-1]["metadata"]["rag"]
    assert len(rag["calls"]) == 1
    fs = rag["calls"][0]["facet_scope"]
    assert fs == {
        "sequence_number": 8,
        "season_number": None,
        "matched_count": 2,
        "no_match": False,
    }


@pytest.mark.asyncio
async def test_rag_metadata_includes_read_source_call(client):
    """#371: read_source calls must appear in metadata.rag.calls alongside find_passages calls."""
    from bibilab.pipeline.chat_tools import READ_SOURCE_TOOL

    list_id = await _create_list(client, "ReadSourceLedger")

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name=READ_SOURCE_TOOL.name, arguments={"source_id": "s1"}),
        )
        result = await execute_tool_fn(READ_SOURCE_TOOL.name, {"source_id": "s1"})
        tool_result_data = {"name": READ_SOURCE_TOOL.name, "result": _client_tool_result(result)}
        yield StreamEvent(type="tool_result", content=json.dumps(tool_result_data))
        yield StreamEvent(type=SSE_EVENT_DELTA, content="ok")
        yield StreamEvent(type=SSE_EVENT_DONE)

    async def fake_execute_tool(**kwargs):
        return {
            "_chunks": "transcript narrative text",
            "source_id": "s1",
        }

    with (
        patch("bibilab.routers.chat.stream_with_tools", fake_stream),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "what is in source s1?"})

    assert resp.status_code == 200

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    rag = assistant_msgs[-1]["metadata"]["rag"]
    assert len(rag["calls"]) == 1
    call = rag["calls"][0]
    assert call["tool_name"] == READ_SOURCE_TOOL.name
    assert call["source_id"] == "s1"
