"""Jobs table SQL CRUD + parse_job_meta row-shape helper.

``reset_stuck_jobs`` (the start-of-day job-state machine flip) lives in
``bibilab.worker`` next to its only caller, so the worker does not import
back into the data layer for a piece of business logic. ``parse_job_meta``
stays here as a row-shape helper for the JSON `meta` column — it pairs
naturally with the row reads and is not a state machine.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import aiosqlite

from bibilab.db.connection import _now, get_db
from bibilab.db.sources import _in_placeholders
from bibilab.models.jobs import JobStatus


def parse_job_meta(job: dict) -> dict[str, Any]:
    """Parse and normalize the meta field from a job row (dict, Row, or JSON string)."""
    row = job if isinstance(job, dict) else dict(job)
    meta = row.get("meta", {}) or {}
    if isinstance(meta, str):
        return json.loads(meta)
    return meta


async def get_jobs_for_video_ids(video_ids: list[str], list_id: str) -> list[dict[str, str]]:
    if not video_ids:
        return []
    placeholders = _in_placeholders(video_ids)

    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT json_extract(meta, '$.video_id') AS video_id, status FROM jobs
            WHERE json_extract(meta, '$.list_id') = ?
            AND json_extract(meta, '$.video_id') IN ({placeholders})
            """,
            [list_id] + video_ids,
        )
        return await cursor.fetchall()


async def get_source_video_ids(video_ids: list[str], list_id: str) -> set[str]:
    if not video_ids:
        return set()
    placeholders = _in_placeholders(video_ids)

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT video_id FROM sources WHERE list_id=? AND video_id IN ({placeholders})",
            [list_id] + video_ids,
        )
        rows = await cursor.fetchall()
    return {row["video_id"] for row in rows}


async def create_job(
    type: str,
    meta: dict[str, Any],
) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO jobs (id, type, status, progress, created_at, updated_at, meta)
            VALUES (?, ?, 'queued', 0, ?, ?, ?)
            """,
            (job_id, type, now, now, json.dumps(meta)),
        )
        await db.commit()
    return job_id


async def update_job_status(
    job_id: str,
    status: str,
    progress: int = 0,
    error: str | None = None,
) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE jobs SET status=?, progress=?, error=?, updated_at=? WHERE id=?",
            (status, progress, error, _now(), job_id),
        )
        await db.commit()


async def update_job_meta(job_id: str, patch: dict) -> None:
    """Merge `patch` into the job's JSON meta column.

    Single-writer assumption: the worker processes jobs sequentially and only
    one coroutine touches a given job at a time, so SELECT-then-UPDATE is safe.
    """
    async with get_db() as db:
        row = await db.execute("SELECT meta FROM jobs WHERE id=?", (job_id,))
        result = await row.fetchone()
        if result is None:
            return
        existing = result[0]
        meta = json.loads(existing) if isinstance(existing, str) else (existing or {})
        meta.update(patch)
        await db.execute(
            "UPDATE jobs SET meta=?, updated_at=? WHERE id=?",
            (json.dumps(meta), _now(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        return await cursor.fetchone()


async def list_jobs() -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        return await cursor.fetchall()


async def delete_job(job_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await db.commit()


async def get_pending_jobs() -> list[aiosqlite.Row]:
    active_statuses = (
        f"'{JobStatus.QUEUED.value}', "
        f"'{JobStatus.DOWNLOADING.value}', "
        f"'{JobStatus.TRANSCRIBING.value}', "
        f"'{JobStatus.PROCESSING.value}'"
    )
    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT * FROM jobs
            WHERE status IN ({active_statuses})
            ORDER BY created_at ASC
            """
        )
        return await cursor.fetchall()
