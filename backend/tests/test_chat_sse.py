"""Tests for the SSE chat streaming endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def an_async_generator(items):
    async def gen():
        for item in items:
            yield item

    return gen()


@pytest.mark.asyncio
async def test_chat_endpoint_returns_sse_stream(client):
    """POST /lists/:id/chat returns text/event-stream with delta events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.return_value = an_async_generator(
            [
                MagicMock(type="delta", content="Hello", tool_call=None),
                MagicMock(type="delta", content=" world", tool_call=None),
                MagicMock(type="done", content=None, tool_call=None),
            ]
        )
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_endpoint_includes_rag_context(client, tmp_bibilab_home):
    """RAG chunks from ChromaDB are included as system parameter to stream_llm."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    captured_system = []

    async def capture_stream(messages, cfg, tools=None, llm_timeout=120, llm_max_tokens=2048, system=None):
        captured_system.append(system)
        yield MagicMock(type="done", content=None, tool_call=None)

    with patch("bibilab.routers.chat.stream_llm", capture_stream):
        with patch("bibilab.routers.chat.retrieve", new_callable=AsyncMock) as mock_retrieve:
            with patch("bibilab.routers.chat.get_sources_for_list", new_callable=AsyncMock) as mock_sources:
                from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk

                mock_sources.return_value = [MagicMock(id="src-1")]
                chunk = RetrievedChunk(
                    content="video transcript here",
                    video_title="Test Video",
                    timestamp_start=10.0,
                    timestamp_end=30.0,
                    video_id="bv123",
                    distance=0.15,
                )
                mock_retrieve.return_value = RetrievalResult(
                    chunks=[chunk],
                    mode="focused",
                    candidates_evaluated=1,
                    sources_with_hits=1,
                    sources_total=1,
                    source_coverage=[],
                )
                resp = await client.post(f"/lists/{list_id}/chat", json={"message": "what is this about?"})

    assert resp.status_code == 200
    assert len(captured_system) == 1
    assert captured_system[0] is not None
    assert "video transcript here" in captured_system[0]


@pytest.mark.asyncio
async def test_chat_endpoint_saves_user_message(client):
    """User message is persisted to DB after successful stream."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.return_value = an_async_generator(
            [
                MagicMock(type="delta", content="Hi"),
                MagicMock(type="done", content=None, tool_call=None),
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

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.return_value = an_async_generator(
            [
                MagicMock(type="delta", content="Hello"),
                MagicMock(type="done", content=None, tool_call=None),
            ]
        )
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    messages = conv_resp.json()["messages"]
    assert any(m["role"] == "assistant" and m["content"] == "Hello" for m in messages)


@pytest.mark.asyncio
async def test_chat_endpoint_404_for_missing_list(client):
    """Returns 404 when list does not exist."""
    resp = await client.post("/lists/nonexistent/chat", json={"message": "hi"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_endpoint_uses_conversation_history(client):
    """Prior conversation messages are included in LLM context."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    from bibilab.db import create_message, get_or_create_conversation

    conv_id = await get_or_create_conversation(list_id)
    await create_message(conv_id, "user", "Previous question", None)
    await create_message(conv_id, "assistant", "Previous answer", None)

    captured_messages = []

    async def capture_stream(messages, cfg, tools=None, llm_timeout=120, llm_max_tokens=2048, system=None):
        captured_messages.append(messages)
        yield MagicMock(type="done", content=None, tool_call=None)

    with patch("bibilab.routers.chat.stream_llm", capture_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "follow-up"})

    assert resp.status_code == 200
    msgs = captured_messages[0]
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles
    contents = [m["content"] for m in msgs]
    assert "Previous question" in contents
    assert "Previous answer" in contents


@pytest.mark.asyncio
async def test_chat_endpoint_includes_tool_definitions(client):
    """Tools from chat_tools are passed to stream_llm."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    captured_tools = []

    async def capture_stream(messages, cfg, tools=None, llm_timeout=120, llm_max_tokens=2048, system=None):
        captured_tools.append(tools)
        yield MagicMock(type="done", content=None, tool_call=None)

    with patch("bibilab.routers.chat.stream_llm", capture_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert captured_tools[0] is not None
    tool_names = [t.name for t in captured_tools[0]]
    assert "generate_report" in tool_names


@pytest.mark.asyncio
async def test_chat_endpoint_handles_tool_call(client):
    """Tool call is executed and result is fed back to LLM for final response."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.name = "generate_report"
    mock_tool_call.arguments = {"type": "brief", "prompt": "summarize"}

    tool_result = {"artifact_id": "abc", "name": "brief", "type": "brief"}

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.side_effect = [
            an_async_generator(
                [
                    MagicMock(type="delta", content="Generating...", tool_call=None),
                    MagicMock(type="tool_call", content=None, tool_call=mock_tool_call),
                ]
            ),
            an_async_generator(
                [
                    MagicMock(type="delta", content="Here is your brief.", tool_call=None),
                    MagicMock(type="done", content=None, tool_call=None),
                ]
            ),
        ]
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = tool_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief"})

    assert resp.status_code == 200
    assert mock_exec.called
    assert "artifact_id" in resp.text or "brief" in resp.text


@pytest.mark.asyncio
async def test_chat_endpoint_saves_tool_call_metadata(client):
    """Assistant message has tool_calls metadata after tool execution."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.name = "generate_report"
    mock_tool_call.arguments = {"type": "brief", "prompt": "summarize"}

    tool_result = {"artifact_id": "abc", "name": "brief", "type": "brief"}

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.side_effect = [
            an_async_generator(
                [
                    MagicMock(type="tool_call", content=None, tool_call=mock_tool_call),
                ]
            ),
            an_async_generator(
                [
                    MagicMock(type="done", content=None, tool_call=None),
                ]
            ),
        ]
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = tool_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    assistant_msgs = [m for m in conv_resp.json()["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["metadata"] is not None
    assert "tool_calls" in assistant_msgs[0]["metadata"]


@pytest.mark.asyncio
async def test_chat_endpoint_yields_tool_result_event(client):
    """SSE stream includes tool_result event after tool execution."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.name = "generate_report"
    mock_tool_call.arguments = {"type": "brief", "prompt": "summarize"}

    tool_result = {"artifact_id": "abc123", "name": "brief", "type": "brief"}

    with patch("bibilab.routers.chat.stream_llm") as mock_stream:
        mock_stream.side_effect = [
            an_async_generator(
                [
                    MagicMock(type="tool_call", content=None, tool_call=mock_tool_call),
                ]
            ),
            an_async_generator(
                [
                    MagicMock(type="done", content=None, tool_call=None),
                ]
            ),
        ]
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = tool_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief"})

    assert resp.status_code == 200
    assert "tool_result" in resp.text
    assert "abc123" in resp.text


@pytest.mark.asyncio
async def test_conversation_mode_defaults_to_focused(client):
    """Conversation is not created on GET; PATCH creates it with focused mode."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    assert conv_resp.json()["conversation"] is None
    resp = await client.patch(f"/lists/{list_id}/conversation", json={"mode": "focused"})
    assert resp.json()["mode"] == "focused"


@pytest.mark.asyncio
async def test_patch_conversation_mode_updates_row(client):
    """PATCH /lists/:id/conversation updates the mode field."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.patch(f"/lists/{list_id}/conversation", json={"mode": "broad"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "broad"
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    assert conv_resp.json()["conversation"]["mode"] == "broad"


@pytest.mark.asyncio
async def test_chat_endpoint_uses_stored_mode(client):
    """Chat endpoint passes stored conversation mode to retrieve()."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    await client.patch(f"/lists/{list_id}/conversation", json={"mode": "broad"})
    captured_mode = []

    async def capture_retrieve(query_text, source_ids, cfg, mode="focused", top_k=10):
        captured_mode.append(mode)
        return MagicMock(
            chunks=[], mode=mode, candidates_evaluated=0, sources_with_hits=0, sources_total=0, source_coverage=[]
        )

    with patch("bibilab.routers.chat.retrieve", capture_retrieve):
        with patch("bibilab.routers.chat.get_sources_for_list", new_callable=AsyncMock) as mock_sources:
            mock_sources.return_value = [MagicMock(id="src-1")]
            with patch("bibilab.routers.chat.stream_llm") as mock_stream:
                mock_stream.return_value = an_async_generator(
                    [
                        MagicMock(type="done", content=None, tool_call=None),
                    ]
                )
                await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})
    assert captured_mode[0] == "broad"


@pytest.mark.asyncio
async def test_rag_meta_event_emitted_before_deltas(client):
    """rag_meta SSE event is emitted before any delta events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk

    with patch("bibilab.routers.chat.retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="test", video_title="V", timestamp_start=0, timestamp_end=1, video_id="v1", distance=0.1
                )
            ],
            mode="focused",
            candidates_evaluated=30,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[],
        )
        with patch("bibilab.routers.chat.get_sources_for_list", new_callable=AsyncMock) as mock_sources:
            mock_sources.return_value = [MagicMock(id="src-1")]
            with patch("bibilab.routers.chat.stream_llm") as mock_stream:
                mock_stream.return_value = an_async_generator(
                    [
                        MagicMock(type="delta", content="Hi", tool_call=None),
                        MagicMock(type="done", content=None, tool_call=None),
                    ]
                )
                resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})
    lines = resp.text.strip().split("\n")
    rag_meta_line = next(line for line in lines if "rag_meta" in line)
    delta_line = next(line for line in lines if "delta" in line)
    assert lines.index(rag_meta_line) < lines.index(delta_line)


@pytest.mark.asyncio
async def test_assistant_message_persisted_with_rag_metadata(client):
    """Assistant message metadata contains rag data after retrieval."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk

    with patch("bibilab.routers.chat.retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    content="test", video_title="V", timestamp_start=0, timestamp_end=1, video_id="v1", distance=0.1
                )
            ],
            mode="focused",
            candidates_evaluated=30,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[],
        )
        with patch("bibilab.routers.chat.get_sources_for_list", new_callable=AsyncMock) as mock_sources:
            mock_sources.return_value = [MagicMock(id="src-1")]
            with patch("bibilab.routers.chat.stream_llm") as mock_stream:
                mock_stream.return_value = an_async_generator(
                    [
                        MagicMock(type="done", content=None, tool_call=None),
                    ]
                )
                await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    msgs = conv_resp.json()["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["metadata"] is not None
    assert assistant["metadata"]["rag"] is not None
    assert assistant["metadata"]["rag"]["mode"] == "focused"
    assert assistant["metadata"]["rag"]["candidates_evaluated"] == 30


@pytest.mark.asyncio
async def test_no_rag_event_when_no_retrieval(client):
    """No rag_meta event when retrieve returns no chunks."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    from bibilab.pipeline.embed import RetrievalResult

    with patch("bibilab.routers.chat.retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = RetrievalResult(
            chunks=[],
            mode="focused",
            candidates_evaluated=0,
            sources_with_hits=0,
            sources_total=0,
            source_coverage=[],
        )
        with patch("bibilab.routers.chat.get_sources_for_list", new_callable=AsyncMock) as mock_sources:
            mock_sources.return_value = [MagicMock(id="src-1")]
            with patch("bibilab.routers.chat.stream_llm") as mock_stream:
                mock_stream.return_value = an_async_generator(
                    [
                        MagicMock(type="done", content=None, tool_call=None),
                    ]
                )
                resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})
    assert "rag_meta" not in resp.text
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    msgs = conv_resp.json()["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["metadata"] is None
