import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiosqlite

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


async def bootstrap_db() -> None:
    async with get_db() as db:
        list_columns = [row[1] for row in await db.execute_fetchall("PRAGMA table_info(lists)")]
        if list_columns and "thumbnail_source_id" not in list_columns:
            thumbnail_select = "thumbnail_source_video_id" if "thumbnail_source_video_id" in list_columns else "NULL"
            await db.execute("ALTER TABLE lists RENAME TO lists_old")
            await db.execute(_CREATE_LISTS)
            await db.execute(
                f"""
                INSERT INTO lists (id, name, thumbnail_source_id, created_at)
                SELECT id, name, {thumbnail_select}, created_at FROM lists_old
                """
            )
            await db.execute("DROP TABLE lists_old")

        job_columns = [row[1] for row in await db.execute_fetchall("PRAGMA table_info(jobs)")]
        if "source_url" in job_columns or "platform" in job_columns:
            await db.execute("ALTER TABLE jobs RENAME TO jobs_old")
            await db.execute(_CREATE_JOBS)
            await db.execute(
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
            await db.execute("DROP TABLE jobs_old")

        # Fresh-start sources migration: drop old table if schema lacks 'id' column
        source_columns = [row[1] for row in await db.execute_fetchall("PRAGMA table_info(sources)")]
        if source_columns and "id" not in source_columns:
            await db.execute("DROP TABLE IF EXISTS sources")
        await db.execute(_CREATE_LISTS)
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_SOURCES)
        await db.execute(_CREATE_ARTIFACTS)
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(get_db_path())
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_list(list_id: str, name: str, created_at: str) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, thumbnail_source_id, created_at) VALUES (?, ?, ?, ?)",
            (list_id, name, None, created_at),
        )
        await db.commit()


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
        cursor = await db.execute("SELECT id FROM sources WHERE video_id = ? AND list_id = ?", (video_id, list_id))
        existing = await cursor.fetchone()

        # Auto-assign thumbnail if new source and list has none
        if existing is None:
            cursor = await db.execute("SELECT thumbnail_source_id FROM lists WHERE id = ?", (list_id,))
            list_row = await cursor.fetchone()
            if list_row is not None and list_row["thumbnail_source_id"] is None:
                await db.execute(
                    "UPDATE lists SET thumbnail_source_id = ? WHERE id = ?",
                    (source_id, list_id),
                )

        # Upsert source using INSERT OR IGNORE + ON CONFLICT DO UPDATE
        await db.execute(
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
        await db.commit()


async def update_source_digest(source_id: str, summary: str, keywords: list[str]) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE sources SET summary=?, keywords=?, processed_at=? WHERE id=?",
            (summary, json.dumps(keywords), _now(), source_id),
        )
        await db.commit()


async def get_source(source_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE id=?", (source_id,))
        return await cursor.fetchone()


async def get_source_by_video_and_list(video_id: str, list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE video_id=? AND list_id=?", (video_id, list_id))
        return await cursor.fetchone()


async def get_sources_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        )
        return await cursor.fetchall()


async def delete_source(source_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        await db.commit()


async def delete_sources_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM sources WHERE list_id=?", (list_id,))
        await db.commit()


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


async def update_artifact_completed(
    artifact_id: str,
    name: str,
    content_path: str,
) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE artifacts SET name=?, content_path=?, status='done' WHERE id=?",
            (name, content_path, artifact_id),
        )
        await db.commit()


async def update_artifact_error(artifact_id: str, error: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE artifacts SET status='failed', error=? WHERE id=?",
            (error, artifact_id),
        )
        await db.commit()


async def delete_artifact(artifact_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
        await db.commit()


async def delete_artifacts_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE list_id=?", (list_id,))
        await db.commit()


async def source_exists(video_id: str, list_id: str) -> bool:
    row = await get_source_by_video_and_list(video_id, list_id)
    return row is not None


async def get_video_statuses(
    video_ids: list[str], list_id: str
) -> dict[str, Literal["new", "processed", "in_progress", "needs_auth"]]:
    if not video_ids:
        return {}
    placeholders = ",".join("?" * len(video_ids))

    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT json_extract(meta, '$.video_id') AS video_id, status FROM jobs
            WHERE json_extract(meta, '$.list_id') = ?
            AND json_extract(meta, '$.video_id') IN ({placeholders})
            """,
            [list_id] + video_ids,
        )
        job_rows = await cursor.fetchall()

        cursor = await db.execute(
            f"SELECT video_id FROM sources WHERE list_id=? AND video_id IN ({placeholders})",
            [list_id] + video_ids,
        )
        source_rows = await cursor.fetchall()

    processed = {row["video_id"] for row in source_rows}
    statuses: dict[str, Literal["new", "processed", "in_progress", "needs_auth"]] = {}

    needs_auth_videos: set[str] = set()
    in_progress_videos: set[str] = set()

    for row in job_rows:
        vid = row["video_id"]
        st = row["status"]
        if st == JobStatus.NEEDS_AUTH.value:
            needs_auth_videos.add(vid)
        elif st in (
            JobStatus.QUEUED.value,
            JobStatus.DOWNLOADING.value,
            JobStatus.TRANSCRIBING.value,
            JobStatus.PROCESSING.value,
            JobStatus.EXTRACTING.value,
            JobStatus.WRITING.value,
        ):
            in_progress_videos.add(vid)

    for vid in video_ids:
        if vid in needs_auth_videos:
            statuses[vid] = "needs_auth"
        elif vid in in_progress_videos:
            statuses[vid] = "in_progress"
        elif vid in processed:
            statuses[vid] = "processed"
        else:
            statuses[vid] = "new"

    return statuses


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


async def reset_stuck_jobs() -> None:
    stuck_statuses = (
        f"'{JobStatus.DOWNLOADING.value}', '{JobStatus.TRANSCRIBING.value}', '{JobStatus.PROCESSING.value}'"
    )
    async with get_db() as db:
        await db.execute(
            f"""
            UPDATE jobs SET status='{JobStatus.QUEUED.value}', updated_at=?
            WHERE status IN ({stuck_statuses})
            """,
            (_now(),),
        )
        await db.commit()
