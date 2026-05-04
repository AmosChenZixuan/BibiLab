"""Tests for the SSE chat streaming endpoint (post-tool-calling refactor)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from bibilab.pipeline._shared import StreamEvent, ToolCall
from bibilab.routers.chat import SSE_EVENT_DELTA, SSE_EVENT_DONE, SSE_EVENT_TOOL_RESULT


def an_async_generator(items):
    async def gen():
        for item in items:
            yield item

    return gen()


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _strip_chunks(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_chunks"}


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

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
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

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
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

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
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

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        result = await execute_tool_fn("retrieve", {"query": "test", "search_mode": "factual"})
        yield StreamEvent(
            type="tool_result",
            content=json.dumps({"name": "retrieve", "result": _strip_chunks(result)}),
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

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
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
                "source_coverage": [{"video_id": "bv1", "title": "Test Video"}],
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
                    {"video_id": "v1", "title": "Video A"},
                    {"video_id": "v2", "title": "Video B"},
                    {"video_id": "v3", "title": "Video C"},
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

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name="retrieve", arguments={"query": query, "search_mode": search_mode}),
        )
        result = await execute_tool_fn("retrieve", {"query": query, "search_mode": search_mode})
        yield StreamEvent(type="tool_result", content=json.dumps({"name": "retrieve", "result": _strip_chunks(result)}))
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
    assert assistant_msgs[0]["metadata"]["rag"]["search_mode"] == search_mode


@pytest.mark.asyncio
async def test_smoke_scenario_3_generate_report_no_retrieve(client):
    """Report request: LLM calls generate_report directly, no retrieve tool involved."""
    list_id = await _create_list(client, "Smoke 3")

    report_result = {"artifact_id": "art-123", "job_id": "job-456", "name": "brief", "type": "brief"}

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
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
