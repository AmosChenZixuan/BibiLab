"""Conversations table CRUD."""

from __future__ import annotations

import uuid

import aiosqlite

from bibilab.db.connection import _now, get_db


async def get_conversation_by_list(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, list_id, summary, active_stream_message_id, created_at, updated_at "
            "FROM conversations WHERE list_id=?",
            (list_id,),
        )
        return await cursor.fetchone()


async def get_or_create_conversation(list_id: str) -> str:
    """Return existing conversation ID for list_id, or create a new one.

    Uses INSERT ... ON CONFLICT to avoid a TOCTOU race between concurrent callers.
    """
    conversation_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO conversations (id, list_id, summary, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?)
            ON CONFLICT(list_id) DO UPDATE SET updated_at=updated_at
            """,
            (conversation_id, list_id, now, now),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE list_id=?",
            (list_id,),
        )
        row = await cursor.fetchone()
        return row["id"]


async def get_conversation(conversation_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, list_id, summary, active_stream_message_id, created_at, updated_at "
            "FROM conversations WHERE id=?",
            (conversation_id,),
        )
        return await cursor.fetchone()


async def delete_conversation(conversation_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        await db.commit()


async def set_active_stream(conversation_id: str, message_id: str | None) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE conversations SET active_stream_message_id=? WHERE id=?",
            (message_id, conversation_id),
        )
        await db.commit()
