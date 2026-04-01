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


_CREATE_LISTS = """
CREATE TABLE IF NOT EXISTS lists (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

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

_CREATE_SOURCES = """
CREATE TABLE IF NOT EXISTS sources (
    video_id            TEXT PRIMARY KEY,
    platform            TEXT NOT NULL,
    list_id             TEXT NOT NULL REFERENCES lists(id),
    title               TEXT NOT NULL DEFAULT '',
    summary             TEXT NOT NULL DEFAULT '',
    note_path           TEXT NOT NULL,
    transcript_path     TEXT,
    whisper_model       TEXT,
    ai_model            TEXT,
    vision_enabled      INTEGER DEFAULT 0,
    processed_at        DATETIME,
    settings_snapshot   TEXT
)
"""


async def bootstrap_db() -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("PRAGMA table_info(lists)") as cur:
            columns = await cur.fetchall()
        if columns:
            await db.execute("ALTER TABLE lists RENAME TO lists_old")
            await db.execute(_CREATE_LISTS)
            await db.execute(
                """
                INSERT INTO lists (id, name, created_at)
                SELECT id, name, created_at FROM lists_old
                """
            )
            await db.execute("DROP TABLE lists_old")
        await db.execute(_CREATE_LISTS)
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_SOURCES)
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        yield db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_list(list_id: str, name: str, created_at: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            (list_id, name, created_at),
        )
        await db.commit()


async def get_all_lists() -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute("SELECT * FROM lists ORDER BY created_at ASC") as cur:
            return await cur.fetchall()


async def get_list(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM lists WHERE id=?", (list_id,)) as cur:
            return await cur.fetchone()


async def delete_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        await db.commit()


async def write_source(
    video_id: str,
    platform: str,
    list_id: str,
    title: str,
    summary: str,
    note_path: str,
    transcript_path: str | None,
    whisper_model: str,
    ai_model: str,
    vision_enabled: bool,
    settings_snapshot: dict[str, Any],
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO sources
                (video_id, platform, list_id, title, summary, note_path,
                 transcript_path, whisper_model, ai_model, vision_enabled,
                 processed_at, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                platform=excluded.platform,
                list_id=excluded.list_id,
                title=excluded.title,
                summary=excluded.summary,
                note_path=excluded.note_path,
                transcript_path=excluded.transcript_path,
                whisper_model=excluded.whisper_model,
                ai_model=excluded.ai_model,
                vision_enabled=excluded.vision_enabled,
                processed_at=excluded.processed_at,
                settings_snapshot=excluded.settings_snapshot
            """,
            (
                video_id,
                platform,
                list_id,
                title,
                summary,
                note_path,
                transcript_path,
                whisper_model,
                ai_model,
                int(vision_enabled),
                _now(),
                json.dumps(settings_snapshot),
            ),
        )
        await db.commit()


async def get_source(video_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM sources WHERE video_id=?", (video_id,)) as cur:
            return await cur.fetchone()


async def get_sources_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        ) as cur:
            return await cur.fetchall()


async def delete_source(video_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM sources WHERE video_id=?", (video_id,))
        await db.commit()


async def delete_sources_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM sources WHERE list_id=?", (list_id,))
        await db.commit()


async def video_is_processed(video_id: str) -> bool:
    row = await get_source(video_id)
    return row is not None


async def get_processed_video_ids(video_ids: list[str]) -> set[str]:
    if not video_ids:
        return set()
    placeholders = ",".join("?" * len(video_ids))
    async with get_db() as db:
        async with db.execute(
            f"SELECT video_id FROM sources WHERE video_id IN ({placeholders})",
            video_ids,
        ) as cur:
            rows = await cur.fetchall()
    return {row["video_id"] for row in rows}


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
            "UPDATE jobs SET status=?, progress=?, error=?, updated_at=? WHERE id=?",
            (status, progress, error, _now(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cur:
            return await cur.fetchone()


async def list_jobs() -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute("SELECT * FROM jobs ORDER BY created_at DESC") as cur:
            return await cur.fetchall()


async def delete_job(job_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await db.commit()


async def get_pending_jobs() -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('queued', 'downloading', 'transcribing', 'extracting', 'writing')
            ORDER BY created_at ASC
            """
        ) as cur:
            return await cur.fetchall()


async def reset_stuck_jobs() -> None:
    async with get_db() as db:
        await db.execute(
            """
            UPDATE jobs SET status='queued', updated_at=?
            WHERE status IN ('downloading', 'transcribing', 'extracting', 'writing')
            """,
            (_now(),),
        )
        await db.commit()
