"""Artifacts table CRUD."""

from __future__ import annotations

import json

import aiosqlite

from bibilab.db.connection import get_db


async def create_artifact(
    artifact_id: str,
    list_id: str,
    name: str | None,
    type: str,
    prompt: str,
    source_ids: list[str],
    status: str,
    content_path: str | None,
    error: str | None = None,
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO artifacts (id, list_id, name, type, prompt, source_ids, status, content_path, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, list_id, name, type, prompt, json.dumps(source_ids), status, content_path, error),
        )
        await db.commit()


async def get_artifact(artifact_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
        return await cursor.fetchone()


async def get_artifacts_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE list_id=? ORDER BY created_at DESC",
            (list_id,),
        )
        return await cursor.fetchall()


async def update_artifact_name(artifact_id: str, name: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE artifacts SET name=? WHERE id=?", (name, artifact_id))
        await db.commit()


async def delete_artifact(artifact_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
        await db.commit()


async def delete_artifacts_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE list_id=?", (list_id,))
        await db.commit()
