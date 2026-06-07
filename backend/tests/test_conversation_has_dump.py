"""Tests for has_dump field on conversation endpoint messages."""

import pytest

from bibilab.db import get_or_create_conversation
from tests.factories import MessageFactory

pytestmark = pytest.mark.integration


async def _create_list(client, name: str) -> str:
    return (await client.post("/lists", json={"name": name})).json()["id"]


async def test_has_dump_true_when_dir_exists(client, tmp_bibilab_home):
    """has_dump is True for a message whose id is a subdirectory of debug/."""
    list_id = await _create_list(client, "dump-exists")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_abc", role="user", content="hi")

    (tmp_bibilab_home / "debug" / "msg_abc").mkdir(parents=True)
    (tmp_bibilab_home / "debug" / "msg_abc" / "call1.json").write_text("{}")

    resp = await client.get(f"/lists/{list_id}/conversation")
    msgs = resp.json()["messages"]
    target = next(m for m in msgs if m["id"] == "msg_abc")
    assert target["has_dump"] is True


async def test_has_dump_false_when_no_dir(client, tmp_bibilab_home):
    """has_dump is False for all messages when debug/ exists but no subdir matches."""
    list_id = await _create_list(client, "dump-missing")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_xyz", role="user", content="hi")

    (tmp_bibilab_home / "debug").mkdir(parents=True)

    resp = await client.get(f"/lists/{list_id}/conversation")
    for m in resp.json()["messages"]:
        assert m["has_dump"] is False


async def test_has_dump_false_when_debug_dir_missing(client, tmp_bibilab_home):
    """has_dump is False for all messages when debug/ does not exist at all."""
    list_id = await _create_list(client, "dump-no-dir")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_qrs", role="user", content="hi")

    # tmp_bibilab_home is fresh — debug/ does not exist
    resp = await client.get(f"/lists/{list_id}/conversation")
    for m in resp.json()["messages"]:
        assert m["has_dump"] is False
