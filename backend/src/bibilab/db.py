import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiosqlite

import bibilab.config
from bibilab.models._enums import ChatMode
from bibilab.models.jobs import JobStatus

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    return bibilab.config.bibilab_home() / "bibilab.db"


def _in_placeholders(ids: list[str]) -> str:
    return ",".join("?" * len(ids)) if ids else ""


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

_CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    video_id UNINDEXED,
    video_title UNINDEXED,
    timestamp_start UNINDEXED,
    timestamp_end UNINDEXED,
    chunk_id UNINDEXED
)
"""

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    list_id    TEXT NOT NULL UNIQUE REFERENCES lists(id) ON DELETE CASCADE,
    summary    TEXT,
    mode       TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    metadata         TEXT,
    created_at       TEXT NOT NULL
)
"""

_CREATE_QUERY_CLASSIFICATIONS = """
CREATE TABLE IF NOT EXISTS query_classifications (
    id            TEXT PRIMARY KEY,
    list_id       TEXT NOT NULL,
    query_text    TEXT NOT NULL,
    query_type    TEXT NOT NULL,
    effective_mode TEXT NOT NULL,
    created_at    TEXT NOT NULL
)
"""


async def bootstrap_db() -> None:
    async with get_db() as db:
        await db.execute(_CREATE_LISTS)
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_SOURCES)
        await db.execute(_CREATE_ARTIFACTS)
        await db.execute(_CREATE_CHUNKS_FTS)
        await db.execute(_CREATE_CONVERSATIONS)
        await db.execute(_CREATE_MESSAGES)
        await db.execute(_CREATE_QUERY_CLASSIFICATIONS)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(get_db_path())
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
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
        cursor = await db.execute("SELECT id FROM sources WHERE video_id = ? AND list_id = ?", (video_id, list_id))
        existing = await cursor.fetchone()

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

        if existing is None:
            cursor = await db.execute("SELECT thumbnail_source_id FROM lists WHERE id = ?", (list_id,))
            list_row = await cursor.fetchone()
            if list_row is not None and list_row["thumbnail_source_id"] is None:
                await db.execute(
                    "UPDATE lists SET thumbnail_source_id = ? WHERE id = ?",
                    (source_id, list_id),
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


async def get_sources_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        )
        return await cursor.fetchall()


async def get_video_ids_for_sources(source_ids: list[str]) -> dict[str, str]:
    """Map source UUIDs to platform video_ids for ChromaDB filtering.
    Returns {source_id: video_id} for each source found.
    Silently returns empty dict if no sources match.
    """
    if not source_ids:
        return {}
    async with get_db() as db:
        placeholders = _in_placeholders(source_ids)
        cursor = await db.execute(
            f"SELECT id, video_id FROM sources WHERE id IN ({placeholders})",
            source_ids,
        )
        rows = await cursor.fetchall()
        return {row["id"]: row["video_id"] for row in rows}


async def delete_source(source_id: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE lists SET thumbnail_source_id = NULL WHERE thumbnail_source_id = ?", (source_id,))
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


async def delete_artifact(artifact_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
        await db.commit()


async def delete_artifacts_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM artifacts WHERE list_id=?", (list_id,))
        await db.commit()


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


def derive_video_statuses(
    video_ids: list[str],
    jobs: list[dict[str, str]],
    processed_ids: set[str],
) -> dict[str, Literal["new", "processed", "in_progress", "needs_auth"]]:
    needs_auth_videos: set[str] = set()
    in_progress_videos: set[str] = set()

    for row in jobs:
        vid = row["video_id"]
        st = row["status"]
        if st == JobStatus.NEEDS_AUTH.value:
            needs_auth_videos.add(vid)
        elif st in (
            JobStatus.QUEUED.value,
            JobStatus.DOWNLOADING.value,
            JobStatus.TRANSCRIBING.value,
            JobStatus.PROCESSING.value,
        ):
            in_progress_videos.add(vid)

    statuses: dict[str, Literal["new", "processed", "in_progress", "needs_auth"]] = {}
    for vid in video_ids:
        if vid in needs_auth_videos:
            statuses[vid] = "needs_auth"
        elif vid in in_progress_videos:
            statuses[vid] = "in_progress"
        elif vid in processed_ids:
            statuses[vid] = "processed"
        else:
            statuses[vid] = "new"

    return statuses


async def get_video_statuses(
    video_ids: list[str], list_id: str
) -> dict[str, Literal["new", "processed", "in_progress", "needs_auth"]]:
    jobs, processed_ids = await asyncio.gather(
        get_jobs_for_video_ids(video_ids, list_id),
        get_source_video_ids(video_ids, list_id),
    )
    return derive_video_statuses(video_ids, jobs, processed_ids)


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
    """Merge `patch` into the job's JSON meta column."""
    import json

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
            (json.dumps(meta, ensure_ascii=False), _now(), job_id),
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


async def create_conversation(list_id: str) -> str:
    conversation_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO conversations (id, list_id, summary, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?)
            """,
            (conversation_id, list_id, now, now),
        )
        await db.commit()
    return conversation_id


async def get_conversation_by_list(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, list_id, summary, mode, created_at, updated_at FROM conversations WHERE list_id=?",
            (list_id,),
        )
        return await cursor.fetchone()


async def get_or_create_conversation(list_id: str) -> str:
    """Return existing conversation ID for list_id, or create a new one.

    Uses INSERT ... ON CONFLICT to avoid a TOCTOU race between concurrent callers.
    """
    conversation_id = str(uuid.uuid4())
    now = _now()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO conversations (id, list_id, summary, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?)
            ON CONFLICT(list_id) DO UPDATE SET updated_at=updated_at
            """,
            (conversation_id, list_id, now, now),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE list_id=?",
            (list_id,),
        )
        row = await cursor.fetchone()
        return row["id"]


async def create_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None,
) -> aiosqlite.Row:
    message_id = str(uuid.uuid4())
    now = _now()
    metadata_json = json.dumps(metadata) if metadata is not None else None
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, metadata_json, now),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM messages WHERE id=?",
            (message_id,),
        )
        return await cursor.fetchone()


async def get_recent_messages(
    conversation_id: str,
    limit: int,
    before_id: str | None = None,
) -> list[aiosqlite.Row]:
    async with get_db() as db:
        if before_id is not None:
            cursor = await db.execute(
                """
                SELECT id, conversation_id, role, content, metadata, created_at FROM messages
                WHERE conversation_id=? AND (created_at, id) < (
                    SELECT created_at, id FROM messages WHERE id=?
                )
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (conversation_id, before_id, limit),
            )
        else:
            cursor = await db.execute(
                """
                SELECT id, conversation_id, role, content, metadata, created_at FROM messages
                WHERE conversation_id=?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
        rows = await cursor.fetchall()
        return list(reversed(rows))


async def update_conversation_summary(conversation_id: str, summary: str) -> None:
    now = _now()
    async with get_db() as db:
        await db.execute(
            "UPDATE conversations SET summary=?, updated_at=? WHERE id=?",
            (summary, now, conversation_id),
        )
        await db.commit()


async def get_message_count(conversation_id: str) -> int:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id=?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return row[0]


async def get_messages_beyond_window(
    conversation_id: str,
    window_size: int,
) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, conversation_id, role, content, metadata, created_at
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS _rn
                FROM messages
                WHERE conversation_id=?
            )
            WHERE _rn > ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id, window_size),
        )
        return list(await cursor.fetchall())


async def delete_messages_by_ids(message_ids: list[str]) -> None:
    if not message_ids:
        return
    async with get_db() as db:
        placeholders = _in_placeholders(message_ids)
        await db.execute(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            message_ids,
        )
        await db.commit()


async def compress_conversation(
    conversation_id: str,
    summary: str,
    message_ids_to_delete: list[str],
) -> None:
    """Atomically update summary and delete old messages in one transaction."""
    now = _now()
    async with get_db() as db:
        await db.execute(
            "UPDATE conversations SET summary=?, updated_at=? WHERE id=?",
            (summary, now, conversation_id),
        )
        if message_ids_to_delete:
            placeholders = _in_placeholders(message_ids_to_delete)
            await db.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                message_ids_to_delete,
            )
        await db.commit()


async def get_conversation(conversation_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, list_id, summary, mode, created_at, updated_at FROM conversations WHERE id=?",
            (conversation_id,),
        )
        return await cursor.fetchone()


async def update_conversation_mode(conversation_id: str, mode: ChatMode) -> None:
    now = _now()
    async with get_db() as db:
        await db.execute(
            "UPDATE conversations SET mode=?, updated_at=? WHERE id=?",
            (mode, now, conversation_id),
        )
        await db.commit()


async def delete_conversation(conversation_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# FTS5 helpers
# ---------------------------------------------------------------------------


async def clear_fts_for_list(list_id: str) -> None:
    """Delete all FTS rows whose video_id belongs to sources in the given list."""
    async with get_db() as db:
        await db.execute(
            "DELETE FROM chunks_fts WHERE video_id IN (SELECT video_id FROM sources WHERE list_id = ?)",
            (list_id,),
        )
        await db.commit()


def _escape_fts_query(query_text: str) -> str:
    """Escape user input for safe FTS5 MATCH evaluation.

    Quotes each whitespace-separated token to disable FTS5 operator parsing
    (`AND`, `OR`, `:`, `*`, `^`) so arbitrary user queries cannot raise
    `OperationalError`. Inner double-quotes are doubled per FTS5 syntax.
    """
    tokens = [t for t in query_text.split() if t]
    if not tokens:
        return ""
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


async def query_fts_rows(
    query_text: str,
    video_ids: list[str],
    top_k: int = 30,
) -> list[aiosqlite.Row]:
    """Run FTS5 MATCH query filtered by video_ids, return rows ranked by BM25.

    Returns an empty list if the query is empty or FTS5 raises a syntax error.
    """
    if not video_ids:
        return []
    match_query = _escape_fts_query(query_text)
    if not match_query:
        return []
    placeholders = _in_placeholders(video_ids)
    async with get_db() as db:
        try:
            cursor = await db.execute(
                f"SELECT content, video_id, video_title, timestamp_start, timestamp_end, rank "
                f"FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? AND video_id IN ({placeholders}) "
                f"ORDER BY rank "
                f"LIMIT ?",
                [match_query, *video_ids, top_k],
            )
            return await cursor.fetchall()
        except aiosqlite.OperationalError as exc:
            logger.warning("FTS5 MATCH query failed (%s); returning empty results", exc)
            return []


async def log_query_classification(
    list_id: str,
    query_text: str,
    query_type: str,
    effective_mode: str,
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO query_classifications (id, list_id, query_text, query_type, effective_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), list_id, query_text, query_type, effective_mode, _now()),
        )
        await db.commit()
