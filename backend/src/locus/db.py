import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from locus.config import LocusConfig, locus_home


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


async def _migrate_legacy_lists(db: aiosqlite.Connection, cfg: LocusConfig) -> None:
    if not cfg.obsidian.vault_path:
        return

    async with db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='lists'") as cur:
        if await cur.fetchone() is None:
            return

    async with db.execute("SELECT id, name, created_at FROM lists ORDER BY created_at ASC") as cur:
        rows = await cur.fetchall()

    from locus.vault import locus_dir, write_list_overview

    for row in rows:
        overview_path = locus_dir(cfg.obsidian) / row["name"] / "_overview.md"
        await asyncio.to_thread(
            write_list_overview,
            overview_path,
            list_id=row["id"],
            list_name=row["name"],
            created_at=row["created_at"] or "",
        )


async def bootstrap_db() -> None:
    from locus.config import load_config

    cfg = load_config()
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_PROCESSING_LOG)
        try:
            await db.execute("ALTER TABLE processing_log ADD COLUMN list_id TEXT")
        except aiosqlite.OperationalError:
            pass
        await _migrate_legacy_lists(db, cfg)
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


async def write_processing_log(
    video_id: str,
    platform: str,
    list_id: str,
    note_path: str,
    transcript_path: str,
    whisper_model: str,
    ai_model: str,
    vision_enabled: bool,
    settings_snapshot: dict[str, Any],
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO processing_log
                (video_id, platform, list_id, note_path, transcript_path,
                 whisper_model, ai_model, vision_enabled,
                 processed_at, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                list_id=excluded.list_id,
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


async def get_processing_log_videos(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM processing_log WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        ) as cur:
            return await cur.fetchall()


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
