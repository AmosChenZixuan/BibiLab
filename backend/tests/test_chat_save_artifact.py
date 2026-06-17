"""Save assistant chat message to artifact (#535).

Covers:
- _load_sources_for_message helper
- _build_chat_message_markdown helper
- POST /api/lists/{list_id}/chat/save-message endpoint (synchronous)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.factories import ConversationFactory, MessageFactory, SourceFactory

pytestmark = pytest.mark.integration


# --- _load_sources_for_message ----------------------------------------------


@pytest.mark.asyncio
async def test_load_sources_no_content_blocks_returns_empty(client: httpx.AsyncClient):
    from bibilab.worker import _load_sources_for_message

    msg = {"metadata": None}
    assert await _load_sources_for_message(msg) == {}


@pytest.mark.asyncio
async def test_load_sources_no_citations_returns_empty(client: httpx.AsyncClient):
    from bibilab.worker import _load_sources_for_message

    msg = {"metadata": {"content_blocks": [{"type": "text", "text": "hello"}]}}
    assert await _load_sources_for_message(msg) == {}


@pytest.mark.asyncio
async def test_load_sources_returns_title_per_source(client: httpx.AsyncClient):
    from bibilab.worker import _load_sources_for_message

    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    src_a = await SourceFactory.build(list_id, title="Episode 2", video_id="BV2")
    src_b = await SourceFactory.build(list_id, title="Episode 5", video_id="BV5")

    msg = {
        "metadata": {
            "content_blocks": [
                {"type": "citation", "index": 1, "source_id": src_a, "section_id": "s1"},
                {"type": "citation", "index": 2, "source_id": src_b, "section_id": "s1"},
            ]
        }
    }
    out = await _load_sources_for_message(msg)
    assert set(out.keys()) == {src_a, src_b}
    assert out[src_a]["title"] == "Episode 2"
    assert out[src_b]["title"] == "Episode 5"


@pytest.mark.asyncio
async def test_load_sources_skips_citations_without_source_id(client: httpx.AsyncClient):
    from bibilab.worker import _load_sources_for_message

    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    src_a = await SourceFactory.build(list_id, title="Episode 2", video_id="BV2")

    msg = {
        "metadata": {
            "content_blocks": [
                {"type": "citation", "index": 1, "source_id": src_a, "section_id": "s1"},
                {"type": "citation", "index": 2, "section_id": "s1"},  # no source_id
            ]
        }
    }
    out = await _load_sources_for_message(msg)
    assert src_a in out


# --- _build_chat_message_markdown --------------------------------------------


def _msg(content: str, content_blocks: list[dict]) -> dict:
    return {"content": content, "metadata": {"content_blocks": content_blocks}}


@pytest.mark.asyncio
async def test_build_markdown_no_citations_returns_prose_verbatim():
    from bibilab.worker import _build_chat_message_markdown

    prose = "The author argues the structure is sound. No citations here."
    msg = _msg(prose, [{"type": "text", "text": prose}])
    out = _build_chat_message_markdown(msg, {})
    assert out == prose


@pytest.mark.asyncio
async def test_build_markdown_appends_references_section():
    from bibilab.worker import _build_chat_message_markdown

    prose = "Section 3 of Episode 2 contains three claims [1], and so does Episode 5 [2]."
    msg = _msg(
        prose,
        [
            {"type": "text", "text": prose},
            {"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": 12.5},
            {"type": "citation", "index": 2, "source_id": "src-b", "timestamp_start": 240.0},
        ],
    )
    sources = {
        "src-a": {"title": "Episode 2"},
        "src-b": {"title": "Episode 5"},
    }
    out = _build_chat_message_markdown(msg, sources)
    assert out.startswith(prose)
    assert "## References" in out
    assert "[1] Episode 2 @ 00:12" in out
    assert "[2] Episode 5 @ 04:00" in out


@pytest.mark.asyncio
async def test_build_markdown_dedupes_same_index():
    from bibilab.worker import _build_chat_message_markdown

    msg = _msg(
        "foo [1] bar [1] baz [1]",
        [
            {"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None},
            {"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None},
        ],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode"}})
    assert out.count("[1] Episode") == 1


@pytest.mark.asyncio
async def test_build_markdown_missing_source_drops_footnote():
    from bibilab.worker import _build_chat_message_markdown

    msg = _msg(
        "foo [1] bar [2]",
        [
            {"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None},
            {"type": "citation", "index": 2, "source_id": "src-missing", "timestamp_start": None},
        ],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode A"}})
    assert "[1] Episode A" in out
    assert "[2]" not in out.split("## References")[1]


@pytest.mark.asyncio
async def test_build_markdown_no_timestamp_omits_time():
    from bibilab.worker import _build_chat_message_markdown

    msg = _msg(
        "foo [1]",
        [{"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None}],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode A"}})
    assert "[1] Episode A" in out
    assert "@" not in out.split("## References")[1]


@pytest.mark.asyncio
async def test_build_markdown_timestamp_past_hour_no_wrap():
    from bibilab.worker import _build_chat_message_markdown

    msg = _msg(
        "foo [1]",
        [{"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": 3725.0}],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Long"}})
    assert "[1] Long @ 62:05" in out


@pytest.mark.asyncio
async def test_build_markdown_zh_header():
    from bibilab.worker import _build_chat_message_markdown

    msg = _msg(
        "foo [1]",
        [{"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None}],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode"}}, lang="zh")
    assert "## 引用来源" in out
    assert "## References" not in out


# --- POST /api/lists/{list_id}/chat/save-message -----------------------------


async def _seed_done_assistant(list_id: str, *, role: str = "assistant", status: str = "done"):
    conv_id = await ConversationFactory.build(list_id)
    row = await MessageFactory.build(conv_id, role=role, content="hi", status=status)
    return row["id"]


@pytest.mark.asyncio
async def test_save_message_returns_artifact(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    src_id = await SourceFactory.build(list_id, title="Episode 2", video_id="BV2")

    conv_id = await ConversationFactory.build(list_id)
    # Pre-seed the user message that triggered the assistant's reply.
    await MessageFactory.build(conv_id, role="user", content="What does the author argue?", status="done")
    metadata = {
        "content_blocks": [
            {"type": "text", "text": "Author argues X."},
            {"type": "citation", "index": 1, "source_id": src_id, "section_id": "s1", "timestamp_start": 12.0},
        ]
    }
    msg_id = (
        await MessageFactory.build(
            conv_id,
            role="assistant",
            status="done",
            content="Author argues X.",
            metadata=metadata,
        )
    )["id"]

    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": msg_id},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "chat_message"
    assert data["status"] == "completed"
    # Name comes from the user prompt, not the assistant prose.
    assert data["name"] == "What does the author argue?"
    # source_ids carries the actual cited sources so the Lab card count is correct.
    assert data["source_ids"] == [src_id]
    assert data["prompt"] == "What does the author argue?"

    # File exists with prose + references.
    content = (tmp_bibilab_home / data["content_path"]).read_text(encoding="utf-8")
    assert "Author argues X." in content
    assert "## References" in content
    assert "[1] Episode 2 @ 00:12" in content


@pytest.mark.asyncio
async def test_save_message_uses_localized_header(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    src_id = await SourceFactory.build(list_id, title="Episode 2", video_id="BV2")
    conv_id = await ConversationFactory.build(list_id)
    await MessageFactory.build(conv_id, role="user", content="Q?", status="done")
    msg_id = (
        await MessageFactory.build(
            conv_id,
            role="assistant",
            status="done",
            content="A.",
            metadata={
                "content_blocks": [
                    {"type": "citation", "index": 1, "source_id": src_id, "section_id": "s1"},
                ]
            },
        )
    )["id"]

    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": msg_id},
        headers={"X-UI-Lang": "zh"},
    )
    assert resp.status_code == 201
    content = (tmp_bibilab_home / resp.json()["content_path"]).read_text(encoding="utf-8")
    assert "## 引用来源" in content
    assert "## References" not in content


@pytest.mark.asyncio
async def test_save_message_list_not_found(client: httpx.AsyncClient):
    resp = await client.post(
        "/lists/nonexistent/chat/save-message",
        json={"message_id": "any"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_save_message_message_not_found(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": "nonexistent"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_save_message_cross_list_returns_404(client: httpx.AsyncClient):
    list_a = (await client.post("/lists", json={"name": "A"})).json()["id"]
    list_b = (await client.post("/lists", json={"name": "B"})).json()["id"]
    msg_id = await _seed_done_assistant(list_a)
    resp = await client.post(
        f"/lists/{list_b}/chat/save-message",
        json={"message_id": msg_id},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_save_message_user_role_returns_422(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    msg_id = await _seed_done_assistant(list_id, role="user")
    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": msg_id},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_status", ["streaming", "failed", "cancelled", "pending"])
async def test_save_message_non_done_status_returns_422(client: httpx.AsyncClient, bad_status: str):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    msg_id = await _seed_done_assistant(list_id, status=bad_status)
    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": msg_id},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_save_message_no_citations_writes_prose_only(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    conv_id = await ConversationFactory.build(list_id)
    msg_id = (
        await MessageFactory.build(
            conv_id,
            role="assistant",
            status="done",
            content="Just prose, no citations.",
            metadata={"content_blocks": [{"type": "text", "text": "Just prose, no citations."}]},
        )
    )["id"]

    resp = await client.post(
        f"/lists/{list_id}/chat/save-message",
        json={"message_id": msg_id},
    )
    assert resp.status_code == 201
    content = (tmp_bibilab_home / resp.json()["content_path"]).read_text(encoding="utf-8")
    assert "Just prose, no citations." in content
    assert "## References" not in content
