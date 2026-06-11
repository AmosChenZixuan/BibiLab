"""Tests for has_dump field on conversation endpoint messages."""

import pytest

from bibilab.config import RagConfig, load_config, save_config
from bibilab.db import get_or_create_conversation
from tests import create_list
from tests.factories import MessageFactory

pytestmark = pytest.mark.integration


def _enable_debug_prompts() -> None:
    cfg = load_config()
    cfg.rag = RagConfig(debug_prompts=True)
    save_config(cfg)


async def test_has_dump_true_when_dir_exists(client, tmp_bibilab_home):
    """has_dump is True for a message whose id matches a flat .json file in debug/."""
    _enable_debug_prompts()
    list_id = await create_list(client, "dump-exists")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_abc", role="user", content="hi")

    (tmp_bibilab_home / "debug").mkdir(parents=True)
    (tmp_bibilab_home / "debug" / "msg_abc.json").write_text("{}")

    resp = await client.get(f"/lists/{list_id}/conversation")
    msgs = resp.json()["messages"]
    target = next(m for m in msgs if m["id"] == "msg_abc")
    assert target["has_dump"] is True


async def test_has_dump_false_when_no_dir(client, tmp_bibilab_home):
    """has_dump is False for all messages when debug/ exists but no matching file."""
    _enable_debug_prompts()
    list_id = await create_list(client, "dump-missing")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_xyz", role="user", content="hi")

    (tmp_bibilab_home / "debug").mkdir(parents=True)

    resp = await client.get(f"/lists/{list_id}/conversation")
    for m in resp.json()["messages"]:
        assert m["has_dump"] is False


async def test_has_dump_false_when_debug_dir_missing(client, tmp_bibilab_home):
    """has_dump is False for all messages when debug/ does not exist at all."""
    _enable_debug_prompts()
    list_id = await create_list(client, "dump-no-dir")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_qrs", role="user", content="hi")

    # tmp_bibilab_home is fresh — debug/ does not exist
    resp = await client.get(f"/lists/{list_id}/conversation")
    for m in resp.json()["messages"]:
        assert m["has_dump"] is False


async def test_has_dump_skips_glob_when_debug_prompts_off(client, tmp_bibilab_home, monkeypatch):
    """With debug_prompts=False (default), the endpoint must not scan the debug dir.

    The default config is debug_prompts=False, so has_dump must always be False
    even if a stale dump file happens to be on disk.
    """
    from bibilab.routers import chat as chat_module

    list_id = await create_list(client, "dump-off")
    conv_id = await get_or_create_conversation(list_id)
    await MessageFactory.build(conv_id, message_id="msg_off", role="user", content="hi")

    # Place a dump file the endpoint should never inspect.
    (tmp_bibilab_home / "debug").mkdir(parents=True)
    (tmp_bibilab_home / "debug" / "msg_off.json").write_text("{}")

    def _explode(*args, **kwargs):
        raise AssertionError("debug_dir.glob must not be called when debug_prompts is off")

    monkeypatch.setattr(chat_module.Path, "glob", _explode)

    resp = await client.get(f"/lists/{list_id}/conversation")
    msgs = resp.json()["messages"]
    target = next(m for m in msgs if m["id"] == "msg_off")
    assert target["has_dump"] is False
