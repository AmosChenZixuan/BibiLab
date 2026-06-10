"""SQLite connection helpers: path resolution, async context manager, schema bootstrap."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

import bibilab.config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_path() -> Path:
    return bibilab.config.bibilab_home() / "bibilab.db"


def source_exists_sync(source_id: str) -> bool:
    """Sync check for a source row's existence (for worker-thread cleanup)."""
    db_path = get_db_path()
    if not db_path.exists():
        return False
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT 1 FROM sources WHERE id = ? LIMIT 1", (source_id,))
        return cur.fetchone() is not None
    finally:
        conn.close()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(get_db_path())
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
    finally:
        await db.close()


# Schema DDL — split across table-specific files for ownership, but bootstrap
# stays centralized so a single PRAGMA journal_mode=WAL finalizes the create
# sequence in one commit.
_CREATE_LISTS = """
CREATE TABLE IF NOT EXISTS lists (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    thumbnail_source_id  TEXT REFERENCES sources(id),
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT,
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
    id                TEXT PRIMARY KEY,
    video_id          TEXT NOT NULL,
    platform          TEXT NOT NULL,
    list_id           TEXT NOT NULL REFERENCES lists(id),
    title             TEXT NOT NULL DEFAULT '',
    cover_url         TEXT,
    source_url        TEXT NOT NULL DEFAULT '',
    duration_seconds  INTEGER NOT NULL DEFAULT 0,
    uploader          TEXT NOT NULL DEFAULT '',
    language          TEXT,
    whisper_model     TEXT,
    ai_model          TEXT,
    processed_at      DATETIME,
    settings_snapshot TEXT,
    series_name       TEXT,
    sequence_number   INTEGER,
    season_number     INTEGER,
    UNIQUE (video_id, list_id)
)
"""

_CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS artifacts (
    id            TEXT PRIMARY KEY,
    list_id       TEXT NOT NULL REFERENCES lists(id),
    name          TEXT,
    type          TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    source_ids    TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'generating',
    content_path  TEXT,
    error         TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    pinyin,
    source_id UNINDEXED,
    video_title UNINDEXED,
    timestamp_start UNINDEXED,
    timestamp_end UNINDEXED,
    chunk_id UNINDEXED,
    seg_start UNINDEXED,
    seg_end UNINDEXED
)
"""

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id                      TEXT PRIMARY KEY,
    list_id                 TEXT NOT NULL UNIQUE REFERENCES lists(id) ON DELETE CASCADE,
    summary                 TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    active_stream_message_id TEXT
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    metadata         TEXT,
    created_at       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'done',
    error            TEXT,
    tool_blocks      TEXT
)
"""

_CREATE_TRANSCRIPT_SEGMENTS = """
CREATE TABLE IF NOT EXISTS transcript_segments (
    id        INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,
    start_s   REAL NOT NULL,
    end_s     REAL NOT NULL,
    speaker   TEXT,
    text      TEXT NOT NULL
)
"""

_CREATE_SECTIONS = """
CREATE TABLE IF NOT EXISTS sections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    seq              INTEGER NOT NULL,
    seg_start        INTEGER NOT NULL,
    seg_end          INTEGER NOT NULL,
    token_count      INTEGER NOT NULL,
    timestamp_start  REAL,
    timestamp_end    REAL,
    summary          TEXT,
    keywords         TEXT
)
"""

_CREATE_SECTIONS_INDEX = "CREATE INDEX IF NOT EXISTS idx_sections_source ON sections(source_id, seq)"


async def bootstrap_db() -> None:
    async with get_db() as db:
        await db.execute(_CREATE_LISTS)
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_SOURCES)
        await db.execute(_CREATE_ARTIFACTS)
        await db.execute(_CREATE_CHUNKS_FTS)
        await db.execute(_CREATE_CONVERSATIONS)
        await db.execute(_CREATE_MESSAGES)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        await db.execute(_CREATE_TRANSCRIPT_SEGMENTS)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_segments_source ON transcript_segments(source_id, seq)")
        await db.execute(_CREATE_SECTIONS)
        await db.execute(_CREATE_SECTIONS_INDEX)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()
