import json
from unittest.mock import patch

import pytest

from bibilab.routers.chat import _client_tool_result
from tests.factories import ConversationFactory, MessageFactory

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_conversations_table_exists(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
        row = await cursor.fetchone()
        assert row is not None

        cursor = await db.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "id" in columns
        assert "list_id" in columns
        assert "summary" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

        cursor = await db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='conversations'")
        sql = (await cursor.fetchone())[0]
        assert "UNIQUE" in sql.upper()


@pytest.mark.asyncio
async def test_messages_table_exists(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        row = await cursor.fetchone()
        assert row is not None

        cursor = await db.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "id" in columns
        assert "conversation_id" in columns
        assert "role" in columns
        assert "content" in columns
        assert "metadata" in columns
        assert "created_at" in columns


@pytest.mark.asyncio
async def test_delete_list_cascades_to_conversation(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        delete_list,
        get_db,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    await delete_list("list-1")

    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM conversations WHERE id=?", (conv_id,))
        assert await cursor.fetchone() is None


@pytest.mark.asyncio
async def test_delete_conversation_cascades_messages(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        delete_conversation,
        get_db,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")
    msg = await MessageFactory.build(
        conv_id,
        content="Hello",
    )

    await delete_conversation(conv_id)

    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM messages WHERE id=?", (msg["id"],))
        assert await cursor.fetchone() is None


@pytest.mark.asyncio
async def test_get_conversation_empty(client):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    resp = await client.get(f"/lists/{list_id}/conversation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation"] is None
    assert data["messages"] == []


@pytest.mark.asyncio
async def test_get_conversation_with_messages(client):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]

    conv_id = await ConversationFactory.build(list_id)
    await MessageFactory.build(
        conv_id,
        content="Hello",
    )
    await MessageFactory.build(
        conv_id,
        role="assistant",
        content="Hi there",
        metadata={"citations": []},
    )
    await MessageFactory.build(
        conv_id,
        content="Tell me more",
    )

    resp = await client.get(f"/lists/{list_id}/conversation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation"] is not None
    assert data["conversation"]["list_id"] == list_id
    assert data["conversation"]["summary"] is None
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hello"
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["metadata"] == {"citations": []}
    assert data["messages"][2]["role"] == "user"
    assert data["messages"][2]["content"] == "Tell me more"


@pytest.mark.asyncio
async def test_get_conversation_pagination(client):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]

    conv_id = await ConversationFactory.build(list_id)
    msgs = []
    for i in range(5):
        msg = await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )
        msgs.append(msg)

    resp = await client.get(f"/lists/{list_id}/conversation?limit=2")
    data = resp.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Message 3"
    assert data["messages"][1]["content"] == "Message 4"

    before = msgs[3]["id"]
    resp = await client.get(f"/lists/{list_id}/conversation?before={before}&limit=2")
    data = resp.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Message 1"
    assert data["messages"][1]["content"] == "Message 2"


@pytest.mark.asyncio
async def test_delete_conversation(client):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]

    conv_id = await ConversationFactory.build(list_id)
    await MessageFactory.build(
        conv_id,
        content="Hello",
    )

    resp = await client.delete(f"/lists/{list_id}/conversation")
    assert resp.status_code == 204

    resp = await client.get(f"/lists/{list_id}/conversation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation"] is None
    assert data["messages"] == []


@pytest.mark.asyncio
async def test_get_conversation_not_found(client):
    resp = await client.get("/lists/nonexistent-list/conversation")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation_not_found(client):
    resp = await client.delete("/lists/nonexistent-list/conversation")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation_no_op(client):
    list_id = (await client.post("/lists", json={"name": "Test List"})).json()["id"]
    resp = await client.delete(f"/lists/{list_id}/conversation")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_new(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, create_list, get_or_create_conversation

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    conv_id = await get_or_create_conversation("list-1")
    assert conv_id is not None


@pytest.mark.asyncio
async def test_get_or_create_conversation_returns_existing(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, create_list, get_or_create_conversation

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    existing_id = await ConversationFactory.build("list-1")
    result_id = await get_or_create_conversation("list-1")
    assert result_id == existing_id


@pytest.mark.asyncio
async def test_chat_persists_section_grained_rag(client, mock_stream_llm):
    """Persisted metadata.rag.calls[0] is section-shaped: section_coverage +
    context[] each carry section_id, not the legacy source_* keys."""
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL

    list_id = (await client.post("/lists", json={"name": "SecGrained"})).json()["id"]
    iteration_count = 0

    async def fake_stream_llm(messages, cfg, tools=None, system=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="tc1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"}),
            )
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    async def fake_execute_tool(**kwargs):
        return {
            "query": kwargs.get("arguments", {}).get("query", ""),
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 1,
            "reranked": False,
            "scoped_pool_size": 1,
            "facet_scope": {
                "sequence_number": None,
                "season_number": None,
                "matched_count": None,
                "no_match": False,
            },
            "section_coverage": [
                {
                    "section_id": "sec-1",
                    "source_id": "s1",
                    "source_title": "Video One",
                    "seq": 1,
                    "timestamp_start": 12.5,
                    "timestamp_end": 60.0,
                }
            ],
            "_chunks": "",
            "_turn_indices": [1],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})
    assert resp.status_code == 200

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    rag = assistant_msgs[-1]["metadata"]["rag"]
    assert len(rag["calls"]) == 1
    call = rag["calls"][0]
    assert call["tool_name"] == "find_passages"
    # section_coverage replaces source_coverage
    assert "section_coverage" in call
    assert "source_coverage" not in call
    assert all("section_id" in s for s in call["section_coverage"])


@pytest.mark.asyncio
async def test_chat_citation_block_carries_section_id_and_timestamp(client, mock_stream_llm):
    """Citation content_blocks include section_id + timestamp_start (T7 → T9 wiring)."""
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL, CitationRegistryEntry
    from bibilab.routers import chat as chat_module

    list_id = (await client.post("/lists", json={"name": "CitSection"})).json()["id"]

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, **kwargs):
        # Manually populate the registry to simulate execute_find_passages having run.
        # The stream yields a citation event with section_id + timestamp_start directly.
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="tc1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"}),
        )
        registry = kwargs.get("registry")
        if registry is not None:
            registry["sec-1"] = CitationRegistryEntry(
                index=1,
                section_id="sec-1",
                source_id="s1",
                title="Video One",
                seq=1,
                citable=True,
                chunk_ids={"s1_12_30"},
                first_chunk_id="s1_12_30",
                timestamp_start=12.5,
                timestamp_end=60.0,
                rerank_score=0.9,
                preview="verbatim text",
            )
        result = await execute_tool_fn(FIND_PASSAGES_TOOL.name, {"query": "q"})
        tool_result_data = {"name": FIND_PASSAGES_TOOL.name, "result": _client_tool_result(result)}
        yield StreamEvent(type="tool_result", content=json.dumps(tool_result_data))
        yield StreamEvent(
            type="citation",
            content=json.dumps(
                {
                    "index": 1,
                    "section_id": "sec-1",
                    "source_id": "s1",
                    "timestamp_start": 12.5,
                    "chunk_ids": ["s1_12_30"],
                }
            ),
        )
        yield StreamEvent(type="delta", content="answer")
        yield StreamEvent(type="done")

    async def fake_execute_tool(**kwargs):
        return {
            "query": "q",
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 1,
            "reranked": False,
            "scoped_pool_size": 1,
            "facet_scope": {
                "sequence_number": None,
                "season_number": None,
                "matched_count": None,
                "no_match": False,
            },
            "section_coverage": [
                {
                    "section_id": "sec-1",
                    "source_id": "s1",
                    "source_title": "Video One",
                    "seq": 1,
                    "timestamp_start": 12.5,
                    "timestamp_end": 60.0,
                }
            ],
            "_chunks": "",
            "_turn_indices": [1],
        }

    with (
        patch.object(chat_module, "stream_with_tools", fake_stream),
        patch.object(chat_module, "execute_tool", fake_execute_tool),
    ):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "q"})
    assert resp.status_code == 200

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    blocks = assistant_msgs[-1]["metadata"]["content_blocks"]
    citation_blocks = [b for b in blocks if b.get("type") == "citation"]
    assert citation_blocks, "no citation content_block persisted"
    cb = citation_blocks[0]
    assert cb["section_id"] == "sec-1"
    assert cb["timestamp_start"] == 12.5
    # context[] is section-keyed
    rag = assistant_msgs[-1]["metadata"]["rag"]
    call = rag["calls"][0]
    assert "context" in call
    assert all("section_id" in c for c in call["context"])


@pytest.mark.asyncio
async def test_coverage_question_retrieves_outline_not_history(client, mock_stream_llm):
    """#396 regression: a coverage question on a non-history episode MUST
    trigger an outline retrieve (find_passages with the matching facet), not
    a history-answered confabulation. Catches:
      - T8 (coverage exemption in grounding prompt): the LLM dispatches
        find_passages(sequence_number=1) instead of answering from history.
      - T5 (facet → outline expansion): the tool result the LLM sees
        carries BOTH section summaries (the outline), not just the hit
        section's excerpts.
    """
    from bibilab.pipeline._shared import StreamEvent, ToolCall
    from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests.factories import SourceFactory

    list_id = (await client.post("/lists", json={"name": "Coverage396"})).json()["id"]

    # 2-section source with sequence_number=1 facet, each section's summary
    # carries a unique marker so the test can assert each showed up in the
    # LLM-facing _chunks.
    segs = [WhisperSegment(start=float(i), end=float(i + 1), text=f"s{i}.", speaker=None) for i in range(20)]
    sections = [
        Section(seg_start=0, seg_end=9, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
        Section(seg_start=10, seg_end=19, token_count=100, timestamp_start=10.0, timestamp_end=20.0),
    ]
    digests = [
        SectionDigest(summary="OUTLINE_MARKER_ALPHA — opening", keywords=["a"]),
        SectionDigest(summary="OUTLINE_MARKER_BETA — closing", keywords=["b"]),
    ]
    await SourceFactory.build(
        list_id,
        video_id="BVcoverage396",
        title="Coverage 396 Video",
        sequence_number=1,
        segments=segs,
        sections=sections,
        section_digests=digests,
    )

    # Turn 1: dispatch the coverage retrieve (T8 invariant — a coverage question
    # must issue find_passages with the matching facet).
    # Turn 2: synthesize from the outline (the LLM-visible _chunks has both
    # summary markers, so a realistic answer cites them).
    iteration_count = 0
    tool_call_seen: list[dict] = []

    async def fake_stream_llm(messages, cfg, tools=None, system=None, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1:
            tool_call = ToolCall(
                id="tc1",
                name=FIND_PASSAGES_TOOL.name,
                arguments={"query": "第1集讲了什么", "sequence_number": 1},
            )
            tool_call_seen.append(tool_call.arguments)
            yield StreamEvent(type="tool_call", tool_call=tool_call)
            yield StreamEvent(type="done")
        else:
            yield StreamEvent(
                type="delta",
                content="OUTLINE_MARKER_ALPHA and OUTLINE_MARKER_BETA together cover the episode.",
            )
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream_llm

    # Synthesize the post-T5 outline-shaped tool result: BOTH section
    # summaries present in _chunks (the outline), and section_coverage lists
    # both sections. The LLM-bound content is _chunks, exactly what the LLM
    # sees on turn 2 — if T5 were broken, _chunks would only carry the hit
    # section and the assertion below would fail.
    async def fake_execute_tool(**kwargs):
        return {
            "query": "第1集讲了什么",
            "tool_name": FIND_PASSAGES_TOOL.name,
            "candidates_evaluated": 1,
            "sources_with_hits": 1,
            "sources_total": 1,
            "reranked": False,
            "scoped_pool_size": 1,
            "facet_scope": {
                "sequence_number": 1,
                "season_number": None,
                "matched_count": 1,
                "no_match": False,
            },
            "section_coverage": [
                {
                    "section_id": "sec-1",
                    "source_id": "s1",
                    "source_title": "Coverage 396 Video",
                    "seq": 1,
                    "timestamp_start": 0.0,
                    "timestamp_end": 10.0,
                },
                {
                    "section_id": "sec-2",
                    "source_id": "s1",
                    "source_title": "Coverage 396 Video",
                    "seq": 2,
                    "timestamp_start": 10.0,
                    "timestamp_end": 20.0,
                },
            ],
            "_chunks": (
                '===== [1] "Coverage 396 Video" · Section 1 =====\n\n'
                "OUTLINE_MARKER_ALPHA — opening\n\n"
                '===== [2] "Coverage 396 Video" · Section 2 =====\n\n'
                "OUTLINE_MARKER_BETA — closing"
            ),
            "_turn_indices": [1, 2],
        }

    with patch("bibilab.routers.chat.execute_tool", fake_execute_tool):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "第1集讲了什么"})
    assert resp.status_code == 200

    # T8 invariant: the LLM dispatched a coverage retrieve with the matching
    # facet, not a history-answered confabulation.
    assert tool_call_seen, "no tool_call was emitted — the LLM confabulated from history"
    assert tool_call_seen[0].get("sequence_number") == 1
    assert tool_call_seen[0].get("query") == "第1集讲了什么"

    conv = (await client.get(f"/lists/{list_id}/conversation")).json()
    assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert assistant_msgs, "no assistant message persisted"
    rag = assistant_msgs[-1]["metadata"]["rag"]
    assert len(rag["calls"]) == 1
    call = rag["calls"][0]
    assert call["tool_name"] == "find_passages"
    # The persisted facet_scope reflects sequence_number=1 (T5/T8 wired end-to-end).
    assert call["facet_scope"]["sequence_number"] == 1
    assert call["facet_scope"]["matched_count"] == 1
    # T5 invariant: BOTH sections of the outline are in the persisted
    # section_coverage (the LLM got the outline, not a partial hit).
    cov = call["section_coverage"]
    assert len(cov) == 2
    assert {row["seq"] for row in cov} == {1, 2}

    # The LLM's answer text references the outline summaries — the only way
    # the synthesis can land both markers is from an outline-shaped _chunks
    # (a single-section hit would have dropped one of the two markers).
    assistant_content = "".join(m["content"] for m in assistant_msgs if m["role"] == "assistant")
    assert "OUTLINE_MARKER_ALPHA" in assistant_content
    assert "OUTLINE_MARKER_BETA" in assistant_content
