"""Save assistant chat message to artifact (#535)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.factories import ConversationFactory, MessageFactory, SourceFactory

pytestmark = pytest.mark.integration


# --- _build_chat_message_markdown --------------------------------------------


def _msg(content: str, content_blocks: list[dict]) -> dict:
    return {"content": content, "metadata": {"content_blocks": content_blocks}}


def test_build_markdown_no_citations_returns_prose_verbatim():
    from bibilab.routers.chat import _build_chat_message_markdown

    prose = "The author argues the structure is sound. No citations here."
    out = _build_chat_message_markdown(_msg(prose, [{"type": "text", "text": prose}]), {})
    assert out == prose


def test_build_markdown_appends_references_section():
    from bibilab.routers.chat import _build_chat_message_markdown

    prose = "Episode 2 says X [1]."
    msg = _msg(
        prose,
        [
            {"type": "text", "text": prose},
            {"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": 12.5},
        ],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode 2"}})
    assert out.startswith(prose)
    assert "## References" in out
    assert "[1] Episode 2 @ 00:12" in out


def test_build_markdown_zh_header():
    from bibilab.routers.chat import _build_chat_message_markdown

    msg = _msg(
        "foo [1]",
        [{"type": "citation", "index": 1, "source_id": "src-a", "timestamp_start": None}],
    )
    out = _build_chat_message_markdown(msg, {"src-a": {"title": "Episode"}}, lang="zh")
    assert "## 引用来源" in out
    assert "## References" not in out


# --- POST /api/lists/{list_id}/chat/save-message -----------------------------


@pytest.mark.asyncio
async def test_save_message_returns_artifact(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    src_id = await SourceFactory.build(list_id, title="Episode 2", video_id="BV2")

    conv_id = await ConversationFactory.build(list_id)
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

    resp = await client.post(f"/lists/{list_id}/chat/save-message", json={"message_id": msg_id})
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "chat_message"
    assert data["status"] == "completed"
    assert data["name"] == "What does the author argue?"
    assert data["source_ids"] == [src_id]
    assert data["prompt"] == "What does the author argue?"

    content = (tmp_bibilab_home / data["content_path"]).read_text(encoding="utf-8")
    assert "Author argues X." in content
    assert "## References" in content
    assert "[1] Episode 2 @ 00:12" in content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_status", "expected_detail"),
    [
        ("list_404", 404, "List not found"),
        ("msg_404", 404, "Message not found"),
        ("user_role", 422, "Only assistant messages"),
        ("not_done", 422, "Message must be done"),
    ],
)
async def test_save_message_validation(
    client: httpx.AsyncClient, case: str, expected_status: int, expected_detail: str
):
    list_id = (await client.post("/lists", json={"name": "L"})).json()["id"]
    if case == "list_404":
        resp = await client.post("/lists/nonexistent/chat/save-message", json={"message_id": "any"})
    elif case == "msg_404":
        resp = await client.post(f"/lists/{list_id}/chat/save-message", json={"message_id": "nonexistent"})
    elif case == "user_role":
        msg_id = await _seed_assistant(list_id, role="user")
        resp = await client.post(f"/lists/{list_id}/chat/save-message", json={"message_id": msg_id})
    else:  # not_done
        msg_id = await _seed_assistant(list_id, status="streaming")
        resp = await client.post(f"/lists/{list_id}/chat/save-message", json={"message_id": msg_id})
    assert resp.status_code == expected_status
    assert expected_detail in resp.json()["detail"]


async def _seed_assistant(list_id: str, *, role: str = "assistant", status: str = "done") -> str:
    conv_id = await ConversationFactory.build(list_id)
    return (await MessageFactory.build(conv_id, role=role, content="hi", status=status))["id"]
