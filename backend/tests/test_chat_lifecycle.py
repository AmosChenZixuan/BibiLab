import asyncio

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_startup_sweep_marks_orphans_failed(tmp_bibilab_home):  # noqa: ARG001
    """Seed DB with messages.status='streaming' rows, run sweep, assert they're fixed."""
    from bibilab.db import bootstrap_db, get_db
    from bibilab.main import sweep_orphaned_streams

    await bootstrap_db()

    conversation_id = "conv-test-sweep"
    list_id = "list-test-sweep"
    now = "2026-01-01T00:00:00"

    async with get_db() as db:
        await db.execute(
            "INSERT OR IGNORE INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            (list_id, "test", now),
        )
        await db.execute(
            "INSERT OR IGNORE INTO conversations "
            "(id, list_id, summary, created_at, updated_at, active_stream_message_id) "
            "VALUES (?, ?, NULL, ?, ?, 'msg-orphan')",
            (conversation_id, list_id, now, now),
        )
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at, status) "
            "VALUES ('msg-orphan', ?, 'assistant', '', NULL, ?, 'streaming')",
            (conversation_id, now),
        )
        await db.commit()

    await sweep_orphaned_streams()

    async with get_db() as db:
        cursor = await db.execute("SELECT status, error FROM messages WHERE id='msg-orphan'")
        msg = await cursor.fetchone()
        assert msg["status"] == "failed"
        assert msg["error"] == "Server restarted during generation"

        cursor = await db.execute("SELECT active_stream_message_id FROM conversations WHERE id=?", (conversation_id,))
        conv = await cursor.fetchone()
        assert conv["active_stream_message_id"] is None

    # Cleanup
    async with get_db() as db:
        await db.execute("DELETE FROM messages WHERE id='msg-orphan'")
        await db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()


@pytest.mark.asyncio
async def test_shutdown_drains_registry():
    """Register a long-running task, cancel it, assert task is cancelled."""
    from bibilab.pipeline.chat_runs import ChatRunRegistry

    reg = ChatRunRegistry()

    async def long_task():
        await asyncio.sleep(60)

    t = asyncio.create_task(long_task())
    msg_id = "msg-shutdown-test"
    reg.register(msg_id, t)

    for mid in reg.all_message_ids():
        reg.cancel(mid)

    with pytest.raises(asyncio.CancelledError):
        await t


@pytest.mark.asyncio
async def test_reattach_404_for_nonexistent_list(client):
    """GET stream against nonexistent list -> 404."""
    response = await client.get("/api/lists/nonexistent-list-id/chat/msg-id/stream")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_reattach_404_message_not_in_list(client, tmp_bibilab_home):  # noqa: ARG001
    """GET stream returns 404 when the message does not belong to the given list."""
    from bibilab.db import get_db

    list_id = "list-reattach-test"
    now = "2026-01-01T00:00:00"
    async with get_db() as db:
        await db.execute(
            "INSERT OR IGNORE INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            (list_id, "test", now),
        )
        await db.commit()

    response = await client.get(f"/api/lists/{list_id}/chat/nonexistent-msg-id/stream")
    assert response.status_code == 404  # message not in this list

    async with get_db() as db:
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()


@pytest.mark.asyncio
async def test_cancel_404_for_nonexistent_message(client, tmp_bibilab_home):  # noqa: ARG001
    """Cancel returns 404 when message does not exist in the given list."""
    from bibilab.db import get_db

    list_id = "list-cancel-test"
    now = "2026-01-01T00:00:00"
    async with get_db() as db:
        await db.execute(
            "INSERT OR IGNORE INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            (list_id, "test", now),
        )
        await db.commit()

    response = await client.post(f"/api/lists/{list_id}/chat/nonexistent-msg-id/cancel")
    assert response.status_code == 404  # message not in this list

    async with get_db() as db:
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()


@pytest.mark.asyncio
async def test_sweep_marks_both_pending_and_streaming_failed(tmp_bibilab_home):  # noqa: ARG001
    """A server-restart sweep must flip BOTH rows of an in-flight turn —
    the user row (status='pending') and the assistant row (status='streaming')
    — to 'failed' so neither leaks into the next conversation replay."""
    from bibilab.db import bootstrap_db, create_list, get_db
    from bibilab.main import sweep_orphaned_streams
    from tests.factories import ConversationFactory, MessageFactory

    await bootstrap_db()
    await create_list("list-sweep-403", "T", "2026-01-01T00:00:00")
    asst_id = "asst-orphan-403"
    conv_id = await ConversationFactory.build("list-sweep-403", active_stream_message_id=asst_id)
    user_row = await MessageFactory.build(
        conv_id, message_id="user-orphan-403", role="user", content="hi", status="pending"
    )
    asst_row = await MessageFactory.build(conv_id, message_id=asst_id, role="assistant", content="", status="streaming")
    user_id, asst_id = user_row["id"], asst_row["id"]

    await sweep_orphaned_streams()

    async with get_db() as db:
        user_status_row = await (await db.execute("SELECT status FROM messages WHERE id=?", (user_id,))).fetchone()
        asst_status_row = await (await db.execute("SELECT status FROM messages WHERE id=?", (asst_id,))).fetchone()
        conv_active_row = await (
            await db.execute("SELECT active_stream_message_id FROM conversations WHERE id=?", (conv_id,))
        ).fetchone()
        assert user_status_row["status"] == "failed", "user(pending) must be swept to failed"
        assert asst_status_row["status"] == "failed", "assistant(streaming) must be swept to failed"
        assert conv_active_row["active_stream_message_id"] is None
