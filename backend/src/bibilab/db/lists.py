"""Lists table CRUD."""

from __future__ import annotations

import aiosqlite

from bibilab.db.connection import get_db

_LIST_DISPLAY_QUERY = """
SELECT
    lists.id,
    lists.name,
    lists.created_at,
    lists.thumbnail_source_id,
    COALESCE(
        (SELECT MAX(processed_at) FROM sources WHERE sources.list_id = lists.id),
        lists.created_at
    ) AS updated_at,
    (SELECT COUNT(*) FROM sources WHERE sources.list_id = lists.id) AS source_count
FROM lists
"""


async def create_list(list_id: str, name: str, created_at: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, thumbnail_source_id, created_at) VALUES (?, ?, ?, ?)",
            (list_id, name, None, created_at),
        )
        await db.commit()


async def get_all_lists() -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(f"{_LIST_DISPLAY_QUERY} ORDER BY updated_at DESC, created_at DESC")
        return await cursor.fetchall()


async def get_list(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM lists WHERE id=?", (list_id,))
        return await cursor.fetchone()


async def get_list_with_display(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(f"{_LIST_DISPLAY_QUERY} WHERE lists.id=?", (list_id,))
        return await cursor.fetchone()


async def delete_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()


async def update_list_name(list_id: str, name: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE lists SET name=? WHERE id=?", (name, list_id))
        await db.commit()


async def update_list_thumbnail(list_id: str, thumbnail_source_id: str | None) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE lists SET thumbnail_source_id=? WHERE id=?",
            (thumbnail_source_id, list_id),
        )
        await db.commit()
