from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

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
            await db.execute(
                "ALTER TABLE processing_log ADD COLUMN list_id TEXT"
            )
        except aiosqlite.OperationalError:
            pass
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        yield db
