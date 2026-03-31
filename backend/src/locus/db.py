import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from locus.config import locus_home


def get_db_path() -> Path:
    return locus_home() / "locus.db"


_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT,
    source_url  TEXT,
    platform    TEXT,
    status      TEXT DEFAULT 'queued',
    progress    INTEGER DEFAULT 0,
    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    meta        TEXT
)
"""

_CREATE_PROCESSING_LOG = """
CREATE TABLE IF NOT EXISTS processing_log (
    video_id            TEXT PRIMARY KEY,
    platform            TEXT,
    list_id             TEXT,
    note_path           TEXT,
    transcript_path     TEXT,
    whisper_model       TEXT,
    ai_model            TEXT,
    vision_enabled      INTEGER DEFAULT 0,
    processed_at        DATETIME,
    settings_snapshot   TEXT
)
"""

_CREATE_LISTS = """
CREATE TABLE IF NOT EXISTS lists (
    id          TEXT PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


async def bootstrap_db() -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_PROCESSING_LOG)
        await db.execute(_CREATE_LISTS)
        # Idempotent migration: add list_id if missing (for existing installs)
        try:
            await db.execute("ALTER TABLE processing_log ADD COLUMN list_id TEXT")
        except aiosqlite.OperationalError:
            pass
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        yield db


# ---------------------------------------------------------------------------
# Job query helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_job(
    type: str,
    source_url: str,
    platform: str,
    meta: dict[str, Any],
) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO jobs (id, type, source_url, platform, status, progress,
                              created_at, updated_at, meta)
            VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?)
            """,
            (job_id, type, source_url, platform, now, now, json.dumps(meta)),
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
            """
            UPDATE jobs SET status=?, progress=?, error=?, updated_at=?
            WHERE id=?
            """,
            (status, progress, error, _now(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cursor:
            return await cursor.fetchone()


async def list_jobs() -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute("SELECT * FROM jobs ORDER BY created_at DESC") as cursor:
            return await cursor.fetchall()


async def delete_job(job_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await db.commit()


async def get_pending_jobs() -> list[aiosqlite.Row]:
    """Return queued jobs and any in-progress jobs left from a prior crash."""
    async with get_db() as db:
        async with db.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('queued', 'downloading', 'transcribing',
                             'extracting', 'writing')
            ORDER BY created_at ASC
            """
        ) as cursor:
            return await cursor.fetchall()


async def reset_stuck_jobs() -> None:
    """On worker startup, reset any in-progress jobs back to queued."""
    async with get_db() as db:
        await db.execute(
            """
            UPDATE jobs SET status='queued', updated_at=?
            WHERE status IN ('downloading', 'transcribing', 'extracting', 'writing')
            """,
            (_now(),),
        )
        await db.commit()
