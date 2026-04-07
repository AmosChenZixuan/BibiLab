import json
import sqlite3
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bibilab.config
from bibilab.models.jobs import JobStatus


def get_db_path() -> Path:
    return bibilab.config.bibilab_home() / "bibilab.db"


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
    summary           TEXT NOT NULL DEFAULT '',
    keywords          TEXT NOT NULL DEFAULT '[]',
    cover_url         TEXT,
    transcript_path   TEXT,
    source_url        TEXT NOT NULL DEFAULT '',
    duration_seconds  INTEGER NOT NULL DEFAULT 0,
    uploader          TEXT NOT NULL DEFAULT '',
    language          TEXT,
    whisper_model     TEXT,
    ai_model          TEXT,
    vision_enabled    INTEGER DEFAULT 0,
    processed_at      DATETIME,
    settings_snapshot TEXT,
    UNIQUE (video_id, list_id)
)
"""


async def bootstrap_db() -> None:
    async with get_db() as db:
        list_columns = [row[1] for row in db.execute("PRAGMA table_info(lists)").fetchall()]
        if list_columns and "thumbnail_source_id" not in list_columns:
            thumbnail_select = "thumbnail_source_video_id" if "thumbnail_source_video_id" in list_columns else "NULL"
            db.execute("ALTER TABLE lists RENAME TO lists_old")
            db.execute(_CREATE_LISTS)
            db.execute(
                f"""
                INSERT INTO lists (id, name, thumbnail_source_id, created_at)
                SELECT id, name, {thumbnail_select}, created_at FROM lists_old
                """
            )
            db.execute("DROP TABLE lists_old")

        job_columns = [row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()]
        if "source_url" in job_columns or "platform" in job_columns:
            db.execute("ALTER TABLE jobs RENAME TO jobs_old")
            db.execute(_CREATE_JOBS)
            db.execute(
                """
                INSERT INTO jobs (id, type, status, progress, error, created_at, updated_at, meta)
                SELECT
                    id,
                    CASE WHEN type = 'video' THEN 'ingest' ELSE type END,
                    status,
                    progress,
                    error,
                    created_at,
                    updated_at,
                    json_set(
                        COALESCE(meta, '{}'),
                        '$.source_url',
                        source_url,
                        '$.platform',
                        platform
                    )
                FROM jobs_old
                """
            )
            db.execute("DROP TABLE jobs_old")

        # Fresh-start sources migration: drop old table if schema lacks 'id' column
        source_columns = [row[1] for row in db.execute("PRAGMA table_info(sources)").fetchall()]
        if source_columns and "id" not in source_columns:
            db.execute("DROP TABLE IF EXISTS sources")
        db.execute(_CREATE_LISTS)
        db.execute(_CREATE_JOBS)
        db.execute(_CREATE_SOURCES)
        db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[sqlite3.Connection, None]:
    db = sqlite3.connect(get_db_path())
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_list(list_id: str, name: str, created_at: str) -> None:
    async with get_db() as db:
        db.execute(
            "INSERT INTO lists (id, name, thumbnail_source_id, created_at) VALUES (?, ?, ?, ?)",
            (list_id, name, None, created_at),
        )
        db.commit()


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


async def get_all_lists() -> list[sqlite3.Row]:
    async with get_db() as db:
        return db.execute(f"{_LIST_DISPLAY_QUERY} ORDER BY updated_at DESC, created_at DESC").fetchall()


async def get_list(list_id: str) -> sqlite3.Row | None:
    async with get_db() as db:
        return db.execute("SELECT * FROM lists WHERE id=?", (list_id,)).fetchone()


async def get_list_with_display(list_id: str) -> sqlite3.Row | None:
    async with get_db() as db:
        return db.execute(f"{_LIST_DISPLAY_QUERY} WHERE lists.id=?", (list_id,)).fetchone()


async def delete_list(list_id: str) -> None:
    async with get_db() as db:
        db.execute("DELETE FROM lists WHERE id=?", (list_id,))
        db.commit()


async def update_list_name(list_id: str, name: str) -> None:
    async with get_db() as db:
        db.execute("UPDATE lists SET name=? WHERE id=?", (name, list_id))
        db.commit()


async def update_list_thumbnail(list_id: str, thumbnail_source_id: str | None) -> None:
    async with get_db() as db:
        db.execute(
            "UPDATE lists SET thumbnail_source_id=? WHERE id=?",
            (thumbnail_source_id, list_id),
        )
        db.commit()


async def write_source(
    source_id: str,
    video_id: str,
    platform: str,
    list_id: str,
    title: str,
    summary: str,
    keywords: list[str],
    cover_url: str | None,
    transcript_path: str | None,
    source_url: str,
    duration_seconds: int,
    uploader: str,
    language: str | None,
    whisper_model: str,
    ai_model: str,
    vision_enabled: bool,
    settings_snapshot: dict[str, Any],
) -> None:
    async with get_db() as db:
        # Check if this is a new source
        existing = db.execute(
            "SELECT id FROM sources WHERE video_id = ? AND list_id = ?", (video_id, list_id)
        ).fetchone()

        # Auto-assign thumbnail if new source and list has none
        if existing is None:
            list_row = db.execute("SELECT thumbnail_source_id FROM lists WHERE id = ?", (list_id,)).fetchone()
            if list_row is not None and list_row["thumbnail_source_id"] is None:
                db.execute(
                    "UPDATE lists SET thumbnail_source_id = ? WHERE id = ?",
                    (source_id, list_id),
                )

        # Upsert source using INSERT OR IGNORE + ON CONFLICT DO UPDATE
        db.execute(
            """
            INSERT INTO sources
                (id, video_id, platform, list_id, title, summary, keywords,
                 cover_url, transcript_path, source_url, duration_seconds, uploader,
                 language, whisper_model, ai_model, vision_enabled,
                 processed_at, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, list_id) DO UPDATE SET
                platform=excluded.platform,
                title=excluded.title,
                summary=excluded.summary,
                keywords=excluded.keywords,
                cover_url=excluded.cover_url,
                transcript_path=excluded.transcript_path,
                source_url=excluded.source_url,
                duration_seconds=excluded.duration_seconds,
                uploader=excluded.uploader,
                language=excluded.language,
                whisper_model=excluded.whisper_model,
                ai_model=excluded.ai_model,
                vision_enabled=excluded.vision_enabled,
                processed_at=excluded.processed_at,
                settings_snapshot=excluded.settings_snapshot
            """,
            (
                source_id,
                video_id,
                platform,
                list_id,
                title,
                summary,
                json.dumps(keywords),
                cover_url,
                transcript_path,
                source_url,
                duration_seconds,
                uploader,
                language,
                whisper_model,
                ai_model,
                int(vision_enabled),
                _now(),
                json.dumps(settings_snapshot),
            ),
        )
        db.commit()


async def update_source_digest(source_id: str, summary: str, keywords: list[str]) -> None:
    async with get_db() as db:
        db.execute(
            "UPDATE sources SET summary=?, keywords=?, processed_at=? WHERE id=?",
            (summary, json.dumps(keywords), _now(), source_id),
        )
        db.commit()


async def get_source(source_id: str) -> sqlite3.Row | None:
    async with get_db() as db:
        return db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()


async def get_source_by_video_and_list(video_id: str, list_id: str) -> sqlite3.Row | None:
    async with get_db() as db:
        return db.execute("SELECT * FROM sources WHERE video_id=? AND list_id=?", (video_id, list_id)).fetchone()


async def get_sources_for_list(list_id: str) -> list[sqlite3.Row]:
    async with get_db() as db:
        return db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        ).fetchall()


async def delete_source(source_id: str) -> None:
    async with get_db() as db:
        db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        db.commit()


async def delete_sources_for_list(list_id: str) -> None:
    async with get_db() as db:
        db.execute("DELETE FROM sources WHERE list_id=?", (list_id,))
        db.commit()


async def source_exists(video_id: str, list_id: str) -> bool:
    row = await get_source_by_video_and_list(video_id, list_id)
    return row is not None


async def get_processed_video_ids(video_ids: list[str], list_id: str) -> set[str]:
    if not video_ids:
        return set()
    placeholders = ",".join("?" * len(video_ids))
    async with get_db() as db:
        rows = db.execute(
            f"SELECT video_id FROM sources WHERE list_id=? AND video_id IN ({placeholders})",
            [list_id] + video_ids,
        ).fetchall()
    return {row["video_id"] for row in rows}


async def create_job(
    type: str,
    meta: dict[str, Any],
) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        db.execute(
            """
            INSERT INTO jobs (id, type, status, progress, created_at, updated_at, meta)
            VALUES (?, ?, 'queued', 0, ?, ?, ?)
            """,
            (job_id, type, now, now, json.dumps(meta)),
        )
        db.commit()
    return job_id


async def update_job_status(
    job_id: str,
    status: str,
    progress: int = 0,
    error: str | None = None,
) -> None:
    async with get_db() as db:
        db.execute(
            "UPDATE jobs SET status=?, progress=?, error=?, updated_at=? WHERE id=?",
            (status, progress, error, _now(), job_id),
        )
        db.commit()


async def get_job(job_id: str) -> sqlite3.Row | None:
    async with get_db() as db:
        return db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


async def list_jobs() -> list[sqlite3.Row]:
    async with get_db() as db:
        return db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()


async def delete_job(job_id: str) -> None:
    async with get_db() as db:
        db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        db.commit()


async def get_pending_jobs() -> list[sqlite3.Row]:
    active_statuses = (
        f"'{JobStatus.QUEUED.value}', "
        f"'{JobStatus.DOWNLOADING.value}', "
        f"'{JobStatus.TRANSCRIBING.value}', "
        f"'{JobStatus.PROCESSING.value}'"
    )
    async with get_db() as db:
        return db.execute(
            f"""
            SELECT * FROM jobs
            WHERE status IN ({active_statuses})
            ORDER BY created_at ASC
            """
        ).fetchall()


async def reset_stuck_jobs() -> None:
    stuck_statuses = (
        f"'{JobStatus.DOWNLOADING.value}', '{JobStatus.TRANSCRIBING.value}', '{JobStatus.PROCESSING.value}'"
    )
    async with get_db() as db:
        db.execute(
            f"""
            UPDATE jobs SET status='{JobStatus.QUEUED.value}', updated_at=?
            WHERE status IN ({stuck_statuses})
            """,
            (_now(),),
        )
        db.commit()
