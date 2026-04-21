import pytest


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
        create_conversation,
        create_list,
        delete_list,
        get_db,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    await delete_list("list-1")

    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM conversations WHERE id=?", (conv_id,))
        assert await cursor.fetchone() is None


@pytest.mark.asyncio
async def test_delete_conversation_cascades_messages(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        delete_conversation,
        get_db,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")
    msg = await create_message(conv_id, "user", "Hello", None)

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
    from bibilab.db import create_conversation, create_message

    conv_id = await create_conversation(list_id)
    await create_message(conv_id, "user", "Hello", None)
    await create_message(conv_id, "assistant", "Hi there", {"citations": []})
    await create_message(conv_id, "user", "Tell me more", None)

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
    from bibilab.db import create_conversation, create_message

    conv_id = await create_conversation(list_id)
    msgs = []
    for i in range(5):
        msg = await create_message(conv_id, "user", f"Message {i}", None)
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
    from bibilab.db import create_conversation, create_message

    conv_id = await create_conversation(list_id)
    await create_message(conv_id, "user", "Hello", None)

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
