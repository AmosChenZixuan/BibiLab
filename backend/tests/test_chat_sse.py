"""Tests for the SSE chat streaming endpoint (post-tool-calling refactor)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from bibilab.pipeline._shared import StreamEvent, ToolCall
from bibilab.routers.chat import (
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_TOOL_RESULT,
    _client_tool_result,
)
from tests import an_async_generator


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

    from bibilab.db import create_message, get_or_create_conversation

    conv_id = await get_or_create_conversation(list_id)
    await create_message(conv_id, "user", "Previous question", None)
    await create_message(conv_id, "assistant", "Previous answer", None)

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
    """Both retrieve and generate_report tools are passed to stream_with_tools."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    captured_tools = []

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        captured_tools.append(tools)
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", capture):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    tool_names = [t.name for t in captured_tools[0]]
    assert "retrieve" in tool_names
    assert "generate_report" in tool_names


@pytest.mark.asyncio
async def test_chat_endpoint_handles_generate_report_tool(client):
    """generate_report tool execution yields proper SSE events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(type="delta", content="Generating...")
        yield StreamEvent(
            type="tool_result",
            content='{"name":"generate_report","result":{"artifact_id":"abc","name":"brief","type":"brief"}}',
        )
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief"})

    assert resp.status_code == 200
    assert "abc" in resp.text
    assert "tool_result" in resp.text


@pytest.mark.asyncio
async def test_retrieve_tool_result_has_coverage(client):
    """retrieve tool_result includes search_mode and coverage metadata."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        result = await execute_tool_fn("retrieve", {"query": "test", "search_mode": "factual"})
        yield StreamEvent(
            type="tool_result",
            content=json.dumps({"name": "retrieve", "result": _client_tool_result("retrieve", result)}),
        )
        yield StreamEvent(type="delta", content="Answer")
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "what is this?"})

    assert resp.status_code == 200
    assert "tool_result" in resp.text
    assert "factual" in resp.text


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
    "search_mode,query,user_message,delta_text,result_overrides",
    [
        (
            "factual",
            "test query",
            "What does the video say about transformers?",
            "Based on the transcript, the answer is...",
            {
                "candidates_evaluated": 30,
                "sources_with_hits": 1,
                "sources_total": 1,
                "source_coverage": [{"source_id": "src-1", "video_id": "bv1", "title": "Test Video"}],
                "_chunks": [{"title": "Test Video", "start": 10.0, "end": 25.0, "content": "relevant chunk"}],
            },
        ),
        (
            "breadth",
            "核心观点",
            "汇总所有视频的核心观点",
            "汇总如下...",
            {
                "candidates_evaluated": 60,
                "sources_with_hits": 3,
                "sources_total": 5,
                "source_coverage": [
                    {"source_id": "src-1", "video_id": "v1", "title": "Video A"},
                    {"source_id": "src-2", "video_id": "v2", "title": "Video B"},
                    {"source_id": "src-3", "video_id": "v3", "title": "Video C"},
                ],
                "_chunks": [],
            },
        ),
    ],
)
async def test_smoke_scenario_2_retrieve(client, search_mode, query, user_message, delta_text, result_overrides):
    """LLM calls retrieve tool with correct search_mode before answering."""
    list_id = await _create_list(client, f"Smoke retrieve {search_mode}")

    retrieve_result = {"search_mode": search_mode, **result_overrides}

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name="retrieve", arguments={"query": query, "search_mode": search_mode}),
        )
        result = await execute_tool_fn("retrieve", {"query": query, "search_mode": search_mode})
        tool_result_data = {"name": "retrieve", "result": _client_tool_result("retrieve", result)}
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
    assert stream_result["name"] == "retrieve"
    assert stream_result["result"]["search_mode"] == search_mode
    assert stream_result["result"]["candidates_evaluated"] == result_overrides["candidates_evaluated"]
    assert "_chunks" not in stream_result["result"], "_chunks must be stripped before sending to client"

    assert SSE_EVENT_DELTA in types
    assert SSE_EVENT_DONE in types

    assistant_msgs = await _get_assistant_msgs(client, list_id)
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["metadata"] is not None
    assert assistant_msgs[0]["metadata"]["rag"]["calls"][0]["search_mode"] == search_mode


@pytest.mark.asyncio
async def test_smoke_scenario_3_generate_report_no_retrieve(client):
    """Report request: LLM calls generate_report directly, no retrieve tool involved."""
    list_id = await _create_list(client, "Smoke 3")

    report_result = {"artifact_id": "art-123", "job_id": "job-456", "name": "brief", "type": "brief"}

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048, **kwargs):
        yield StreamEvent(type=SSE_EVENT_DELTA, content="Let me generate a brief for you.")
        result = await execute_tool_fn("generate_report", {"type": "brief", "prompt": "summarize the videos"})
        yield StreamEvent(type="tool_result", content=json.dumps({"name": "generate_report", "result": result}))
        yield StreamEvent(type=SSE_EVENT_DONE)

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = report_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief summarizing all videos"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]

    assert SSE_EVENT_TOOL_RESULT in types
    tool_results = [e for e in events if e["type"] == SSE_EVENT_TOOL_RESULT]
    report_events = [tr for tr in tool_results if tr.get("name") == "generate_report"]
    retrieve_events = [tr for tr in tool_results if tr.get("name") == "retrieve"]
    assert len(report_events) >= 1, "generate_report tool_result must be present"
    assert len(retrieve_events) == 0, "generate_report should NOT trigger retrieve"

    assert report_events[0]["result"]["artifact_id"] == "art-123"
    assert report_events[0]["result"]["type"] == "brief"
    assert SSE_EVENT_DELTA in types
    assert SSE_EVENT_DONE in types


@pytest.mark.asyncio
async def test_query_list_metadata_in_loopback_tools():
    from bibilab.routers.chat import LOOPBACK_TOOLS

    assert "query_list_metadata" in LOOPBACK_TOOLS


@pytest.mark.asyncio
async def test_query_list_metadata_tool_registered_for_chat(client):
    """Behavioral: the chat endpoint must pass QUERY_LIST_METADATA_TOOL to stream_llm."""
    from bibilab.pipeline.chat_tools import (
        GENERATE_REPORT_TOOL,
        QUERY_LIST_METADATA_TOOL,
        RETRIEVE_TOOL,
    )
    from bibilab.routers.chat import SSE_EVENT_DELTA, SSE_EVENT_DONE

    captured_tools = None

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal captured_tools
        captured_tools = tools
        # Yield one delta so the loop doesn't hang waiting for done
        yield type("StreamEvent", (), {"type": SSE_EVENT_DELTA, "content": "hi"})()
        yield type("StreamEvent", (), {"type": SSE_EVENT_DONE})()

    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_llm", fake_stream_llm):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert captured_tools is not None, "stream_llm was never called"
    tool_names = {t.name for t in captured_tools}
    assert tool_names == {RETRIEVE_TOOL.name, QUERY_LIST_METADATA_TOOL.name, GENERATE_REPORT_TOOL.name}


def test_grounding_prompt_routes_counts_to_metadata_tool():
    from bibilab.routers.chat import GROUNDING_SYSTEM_PROMPT

    # The old phrasing ("counts across sources" → retrieve) must be gone.
    assert "counts across sources" not in GROUNDING_SYSTEM_PROMPT
    # New routing must mention the metadata tool by name for the LLM.
    assert "query_list_metadata" in GROUNDING_SYSTEM_PROMPT


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
async def test_chat_sse_multi_retrieve_no_crash(client):
    """Smoke test: two retrieve calls in one turn do not crash the SSE stream.

    The registry ordering and dedup logic is exercised in
    test_citation_registry.py; this test verifies the SSE layer
    handles the multi-call flow without internal errors.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_with_tools(*args, **kwargs):
        tc1 = ToolCall(id="tc1", name="retrieve", arguments={"query": "q1", "search_mode": "factual"})
        tc2 = ToolCall(id="tc2", name="retrieve", arguments={"query": "q2", "search_mode": "factual"})
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
async def test_retrieve_filtered_after_first_use(client):
    """After retrieve runs in iteration 1, iteration 2's tools list excludes retrieve."""
    from bibilab.pipeline.chat_tools import (
        GENERATE_REPORT_TOOL,
        QUERY_LIST_METADATA_TOOL,
        RETRIEVE_TOOL,
    )

    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    captured_tools_per_iteration: list[list[str]] = []
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        captured_tools_per_iteration.append([t.name for t in (tools or [])])
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name="retrieve", arguments={"query": "x", "search_mode": "factual"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        return {
            "query": kwargs.get("arguments", {}).get("query", ""),
            "search_mode": "factual",
            "candidates_evaluated": 0,
            "sources_with_hits": 0,
            "sources_total": 1,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with (
        patch("bibilab.routers.chat.stream_llm", fake_stream_llm),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    assert iteration_count == 2
    assert RETRIEVE_TOOL.name in captured_tools_per_iteration[0]
    assert RETRIEVE_TOOL.name not in captured_tools_per_iteration[1]
    # Other tools remain available
    assert QUERY_LIST_METADATA_TOOL.name in captured_tools_per_iteration[1]
    assert GENERATE_REPORT_TOOL.name in captured_tools_per_iteration[1]


@pytest.mark.asyncio
async def test_metadata_then_retrieve_allowed(client):
    """query_list_metadata in iteration 1 must NOT block retrieve in iteration 2."""
    from bibilab.pipeline.chat_tools import (
        QUERY_LIST_METADATA_TOOL,
        RETRIEVE_TOOL,
    )

    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    captured_tools_per_iteration: list[list[str]] = []
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        captured_tools_per_iteration.append([t.name for t in (tools or [])])
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name="query_list_metadata", arguments={"query_type": "count"}),
            )
            yield StreamEvent(type="done")
        elif iteration_count == 2:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc2", name="retrieve", arguments={"query": "x", "search_mode": "factual"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        name = kwargs.get("name", "")
        args = kwargs.get("arguments", {})
        if name == "query_list_metadata":
            return {"count": 8}
        return {
            "query": args.get("query", ""),
            "search_mode": "factual",
            "candidates_evaluated": 0,
            "sources_with_hits": 0,
            "sources_total": 1,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with (
        patch("bibilab.routers.chat.stream_llm", fake_stream_llm),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    assert iteration_count == 3
    # Iteration 1: all tools available including retrieve
    assert RETRIEVE_TOOL.name in captured_tools_per_iteration[0]
    assert QUERY_LIST_METADATA_TOOL.name in captured_tools_per_iteration[0]
    # Iteration 2: retrieve is still available — metadata didn't consume the allowance
    assert RETRIEVE_TOOL.name in captured_tools_per_iteration[1]
    assert "answer" in resp.text


@pytest.mark.asyncio
async def test_preamble_suppressed_for_tool_iteration(client):
    """Deltas from an iteration that ends with a retrieve tool_call must not reach the client."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(type="delta", content="Let me look that up again.")
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name="retrieve", arguments={"query": "x", "search_mode": "factual"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="final answer")
            yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "search_mode": "factual",
            "candidates_evaluated": 0,
            "sources_with_hits": 0,
            "sources_total": 1,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with (
        patch("bibilab.routers.chat.stream_llm", fake_stream_llm),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    body = resp.text
    assert "Let me look that up again" not in body, "preamble from tool-emitting iteration leaked to client"
    assert "final answer" in body, "terminal iteration's delta must reach the client"


@pytest.mark.asyncio
async def test_terminal_iteration_deltas_streamed(client):
    """A no-tool iteration's deltas must reach the client verbatim."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        yield StreamEvent(type="delta", content="hello world")
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_llm", fake_stream_llm):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "hello world" in resp.text


@pytest.mark.asyncio
async def test_chat_sse_hallucinated_index_emitted_as_text(client, caplog):
    """[7] when registry only has 1-2 → emitted as text delta, warning logged.

    Mocks stream_llm (not stream_with_tools) so the real parser runs.
    The registry stays empty (no retrieve triggered), so [7] is out of range
    and the parser logs citation_hallucinated_index.
    """
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]

    async def fake_stream_llm(*args, **kwargs):
        yield StreamEvent(type="delta", content="see [7]")
        yield StreamEvent(type="done")

    import logging

    with patch("bibilab.routers.chat.stream_llm", side_effect=fake_stream_llm):
        with caplog.at_level(logging.WARNING, logger="bibilab.pipeline.citation_parser"):
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert all(e["type"] == "delta" or e["type"] == "done" for e in events)
    delta_text = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "[7]" in delta_text
    assert any("citation_hallucinated_index" in rec.message for rec in caplog.records)


def test_grounding_prompt_has_audit_prior_claims_rule():
    from bibilab.routers.chat import GROUNDING_SYSTEM_PROMPT

    assert "audit ALL items" in GROUNDING_SYSTEM_PROMPT, (
        "GROUNDING_SYSTEM_PROMPT must instruct the LLM to audit all prior claims, "
        "not just user-flagged ones, when correcting a previous answer."
    )


@pytest.mark.asyncio
async def test_tool_call_start_emitted_before_tool_result(client):
    """Each retrieve tool_call must emit a tool_call_start SSE event before its tool_result."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name="retrieve", arguments={"query": "noodles", "search_mode": "breadth"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "search_mode": args.get("search_mode"),
            "candidates_evaluated": 5,
            "sources_with_hits": 2,
            "sources_total": 3,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with (
        patch("bibilab.routers.chat.stream_llm", fake_stream_llm),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})

    assert resp.status_code == 200
    body = resp.text
    start_idx = body.find('"tool_call_start"')
    result_idx = body.find('"tool_result"')
    assert start_idx >= 0, "tool_call_start event missing"
    assert result_idx >= 0, "tool_result event missing"
    assert start_idx < result_idx, "tool_call_start must precede tool_result"
    assert '"name": "retrieve"' in body
    assert '"query": "noodles"' in body
    assert '"search_mode": "breadth"' in body


@pytest.mark.asyncio
async def test_rag_metadata_persists_calls_list(client):
    """metadata.rag.calls must contain one entry per retrieve call in the turn."""
    list_id = (await client.post("/lists", json={"name": "T"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name="retrieve", arguments={"query": "A", "search_mode": "breadth"}),
            )
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc2", name="retrieve", arguments={"query": "B", "search_mode": "factual"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        args = kwargs.get("arguments", {})
        return {
            "query": args.get("query", ""),
            "search_mode": args.get("search_mode"),
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 5,
            "source_coverage": [],
            "_chunks": "",
            "_turn_indices": [],
        }

    with (
        patch("bibilab.routers.chat.stream_llm", fake_stream_llm),
        patch("bibilab.routers.chat.execute_tool", fake_execute_tool),
    ):
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
    modes = sorted(c["search_mode"] for c in rag["calls"])
    assert modes == ["breadth", "factual"]
