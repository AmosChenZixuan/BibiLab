import json
import logging
import re
import sqlite3
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
from pypinyin import Style, lazy_pinyin

import bibilab.config
from bibilab.models.jobs import JobStatus

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    return bibilab.config.bibilab_home() / "bibilab.db"


def source_exists_sync(source_id: str) -> bool:
    """Sync check for a source row's existence (for worker-thread cleanup)."""
    db_path = get_db_path()
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT 1 FROM sources WHERE id = ? LIMIT 1", (source_id,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def _in_placeholders(ids: list[str]) -> str:
    return ",".join("?" * len(ids)) if ids else ""


def parse_job_meta(job: dict) -> dict[str, Any]:
    """Parse and normalize the meta field from a job row (dict, Row, or JSON string)."""
    row = job if isinstance(job, dict) else dict(job)
    meta = row.get("meta", {}) or {}
    if isinstance(meta, str):
        return json.loads(meta)
    return meta


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
    source_url        TEXT NOT NULL DEFAULT '',
    duration_seconds  INTEGER NOT NULL DEFAULT 0,
    uploader          TEXT NOT NULL DEFAULT '',
    language          TEXT,
    whisper_model     TEXT,
    ai_model          TEXT,
    vision_enabled    INTEGER DEFAULT 0,
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


async def _exec_write_source(
    db: aiosqlite.Connection,
    *,
    source_id: str,
    video_id: str,
    platform: str,
    list_id: str,
    title: str,
    summary: str,
    keywords: list[str],
    cover_url: str | None,
    source_url: str,
    duration_seconds: int,
    uploader: str,
    language: str | None,
    whisper_model: str,
    ai_model: str,
    vision_enabled: bool,
    settings_snapshot: dict[str, Any],
    series_name: str | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
) -> None:
    """Upsert a source row on the given connection. Does NOT commit — the caller
    owns the transaction (lets a source + its segments commit atomically)."""
    cursor = await db.execute("SELECT id FROM sources WHERE video_id = ? AND list_id = ?", (video_id, list_id))
    existing = await cursor.fetchone()

    await db.execute(
        """
        INSERT INTO sources
            (id, video_id, platform, list_id, title, summary, keywords,
             cover_url, source_url, duration_seconds, uploader,
             language, whisper_model, ai_model, vision_enabled,
             processed_at, settings_snapshot,
             series_name, sequence_number, season_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, list_id) DO UPDATE SET
            platform=excluded.platform,
            title=excluded.title,
            summary=excluded.summary,
            keywords=excluded.keywords,
            cover_url=excluded.cover_url,
            source_url=excluded.source_url,
            duration_seconds=excluded.duration_seconds,
            uploader=excluded.uploader,
            language=excluded.language,
            whisper_model=excluded.whisper_model,
            ai_model=excluded.ai_model,
            vision_enabled=excluded.vision_enabled,
            processed_at=excluded.processed_at,
            settings_snapshot=excluded.settings_snapshot,
            series_name=COALESCE(excluded.series_name, series_name),
            sequence_number=COALESCE(excluded.sequence_number, sequence_number),
            season_number=COALESCE(excluded.season_number, season_number)
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
            source_url,
            duration_seconds,
            uploader,
            language,
            whisper_model,
            ai_model,
            int(vision_enabled),
            _now(),
            json.dumps(settings_snapshot),
            series_name,
            sequence_number,
            season_number,
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


async def _exec_write_transcript_segments(db: aiosqlite.Connection, source_id: str, segments: list) -> None:
    """Replace a source's transcript segments on the given connection. Does NOT
    commit. `segments` is a list of WhisperSegment (start, end, text, speaker)."""
    await db.execute("DELETE FROM transcript_segments WHERE source_id = ?", (source_id,))
    await db.executemany(
        "INSERT INTO transcript_segments (source_id, seq, start_s, end_s, speaker, text) VALUES (?, ?, ?, ?, ?, ?)",
        [(source_id, i, s.start, s.end, s.speaker, s.text) for i, s in enumerate(segments)],
    )


async def write_source_with_segments(*, segments: list, **source_fields: Any) -> None:
    """Atomically upsert a source row and its transcript segments in one
    transaction. Either both land or neither — no orphaned source row, no
    compensating delete. `source_fields` are the keyword args of `write_source`.

    On any failure the exception propagates and the connection closes without a
    commit, so SQLite rolls the whole transaction back. FK holds within the
    transaction: the uncommitted parent source row is visible to the child
    segment INSERTs on the same connection."""
    async with get_db() as db:
        await _exec_write_source(db, **source_fields)
        await _exec_write_transcript_segments(db, source_fields["source_id"], segments)
        await db.commit()


async def update_source_digest(
    source_id: str,
    summary: str,
    keywords: list[str],
    series_name: str | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
    bump_processed_at: bool = True,
) -> None:
    # Reruns pass bump_processed_at=False: processed_at anchors list ordering
    # (ORDER BY processed_at ASC in list_sources), and a digest rerun shouldn't
    # move the source within the list.
    sets = [
        "summary=?",
        "keywords=?",
        "series_name=COALESCE(?, series_name)",
        "sequence_number=COALESCE(?, sequence_number)",
        "season_number=COALESCE(?, season_number)",
    ]
    params: list[object] = [summary, json.dumps(keywords), series_name, sequence_number, season_number]
    if bump_processed_at:
        sets.append("processed_at=?")
        params.append(_now())
    params.append(source_id)
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE sources SET {', '.join(sets)} WHERE id=?",
            params,
        )
        if cursor.rowcount == 0:
            raise LookupError(source_id)
        await db.commit()


# Manual-edit facet writer. Replace semantics (explicit None clears), distinct
# from update_source_digest's COALESCE-preserve.
_FACET_WRITE_COLUMNS = ("series_name", "sequence_number", "season_number")


async def update_source_facets(source_id: str, **fields: object) -> None:
    cols = [c for c in _FACET_WRITE_COLUMNS if c in fields]
    if not cols:
        return
    # Column names come from the fixed allowlist above (never user input);
    # values stay parameterized — db.py's no-f-string-values rule holds.
    set_clause = ", ".join(f"{c}=?" for c in cols)
    params = [fields[c] for c in cols]
    params.append(source_id)
    async with get_db() as db:
        cursor = await db.execute(f"UPDATE sources SET {set_clause} WHERE id=?", params)
        if cursor.rowcount == 0:
            # Source vanished between the router's existence check and this
            # write (TOCTOU). Don't commit a no-op as success.
            raise LookupError(source_id)
        await db.commit()


async def get_source(source_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sources WHERE id=?", (source_id,))
        return await cursor.fetchone()


async def write_transcript_segments(source_id: str, segments: list) -> None:
    """Replace all transcript segments for a source. `segments` is a list of
    WhisperSegment (start, end, text, speaker). Idempotent (DELETE then INSERT).

    Standalone segment-write primitive. Production writes segments atomically
    with the source via `write_source_with_segments`; this is the unit-level
    seam the segments-table tests exercise as their subject (round-trip,
    cascade-on-delete, FK orphan rejection) — keep it even with zero prod
    callers, those tests can't express the orphan case via the atomic path."""
    async with get_db() as db:
        await _exec_write_transcript_segments(db, source_id, segments)
        await db.commit()


async def get_transcript_segments(source_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT seq, start_s, end_s, speaker, text FROM transcript_segments WHERE source_id = ? ORDER BY seq",
            (source_id,),
        )
        return await cursor.fetchall()


async def get_segments_for_ranges(ranges: list[tuple[str, int, int]]) -> list[aiosqlite.Row]:
    """Fetch transcript segments for many (source_id, seq_start, seq_end) ranges in one query.

    Used by chat top-k reconstruction: each retained chunk contributes one
    range; the union is fetched once and sliced per chunk by the caller.
    """
    if not ranges:
        return []
    clauses = " OR ".join("(source_id = ? AND seq BETWEEN ? AND ?)" for _ in ranges)
    params: list = [val for r in ranges for val in r]
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT source_id, seq, start_s, end_s, speaker, text "
            f"FROM transcript_segments WHERE {clauses} ORDER BY source_id, seq",
            params,
        )
        return await cursor.fetchall()


async def get_sources_for_list(list_id: str) -> list[aiosqlite.Row]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM sources WHERE list_id=? ORDER BY processed_at ASC",
            (list_id,),
        )
        return await cursor.fetchall()


async def get_source_facets(source_ids: list[str]) -> dict[str, dict[str, int | None]]:
    """Return {source_id: {"sequence_number": int|None, "season_number": int|None}}.

    Only sources found in the table are included. Empty input → {}.
    series_name is intentionally not returned (#309: fuzzy string match deferred).
    """
    if not source_ids:
        return {}
    async with get_db() as db:
        placeholders = _in_placeholders(source_ids)
        cursor = await db.execute(
            f"SELECT id, sequence_number, season_number FROM sources WHERE id IN ({placeholders})",
            source_ids,
        )
        rows = await cursor.fetchall()
        return {
            row["id"]: {
                "sequence_number": row["sequence_number"],
                "season_number": row["season_number"],
            }
            for row in rows
        }


async def delete_source(source_id: str) -> None:
    async with get_db() as db:
        await db.execute("UPDATE lists SET thumbnail_source_id = NULL WHERE thumbnail_source_id = ?", (source_id,))
        await db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        await db.commit()


async def delete_sources_for_list(list_id: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE lists SET thumbnail_source_id = NULL WHERE id = ? AND thumbnail_source_id IN "
            "(SELECT id FROM sources WHERE list_id = ?)",
            (list_id, list_id),
        )
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


async def get_conversation_by_list(list_id: str) -> aiosqlite.Row | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, list_id, summary, active_stream_message_id, created_at, updated_at "
            "FROM conversations WHERE list_id=?",
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


async def get_recent_messages(
    conversation_id: str,
    limit: int,
    before_id: str | None = None,
) -> list[aiosqlite.Row]:
    async with get_db() as db:
        if before_id is not None:
            cursor = await db.execute(
                """
                SELECT id, conversation_id, role, content, metadata,
                       created_at, status, error, tool_blocks
                FROM messages
                WHERE conversation_id=? AND (created_at, rowid) < (
                    SELECT created_at, rowid FROM messages WHERE id=?
                )
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (conversation_id, before_id, limit),
            )
        else:
            cursor = await db.execute(
                """
                SELECT id, conversation_id, role, content, metadata,
                       created_at, status, error, tool_blocks
                FROM messages
                WHERE conversation_id=?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
        rows = await cursor.fetchall()
        return list(reversed(rows))


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
            SELECT id, conversation_id, role, content, metadata, created_at, status, error, tool_blocks
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY created_at DESC, rowid DESC) AS _rn
                FROM messages
                WHERE conversation_id=?
            )
            WHERE _rn > ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id, window_size),
        )
        return list(await cursor.fetchall())


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
            "SELECT id, list_id, summary, active_stream_message_id, created_at, updated_at "
            "FROM conversations WHERE id=?",
            (conversation_id,),
        )
        return await cursor.fetchone()


async def delete_conversation(conversation_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        await db.commit()


class ActiveStreamConflict(Exception):
    """Raised when attempting to start a second stream on an active conversation."""


async def create_user_and_assistant_atomic(
    conversation_id: str,
    user_msg_id: str,
    assistant_msg_id: str,
    user_text: str,
) -> None:
    """Insert user + streaming assistant message atomically, set active_stream_message_id.

    Uses BEGIN IMMEDIATE to serialize concurrent callers — the second caller
    sees active_stream_message_id != NULL and gets ActiveStreamConflict.
    """
    now = _now()
    async with get_db() as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await db.execute(
                "SELECT active_stream_message_id FROM conversations WHERE id=?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row and row["active_stream_message_id"] is not None:
                raise ActiveStreamConflict(
                    f"Conversation {conversation_id} already has active stream {row['active_stream_message_id']}"
                )

            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at, status) "
                "VALUES (?, ?, 'user', ?, NULL, ?, 'done')",
                (user_msg_id, conversation_id, user_text, now),
            )
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at, status) "
                "VALUES (?, ?, 'assistant', '', NULL, ?, 'streaming')",
                (assistant_msg_id, conversation_id, now),
            )
            await db.execute(
                "UPDATE conversations SET active_stream_message_id=? WHERE id=?",
                (assistant_msg_id, conversation_id),
            )
            await db.commit()
        except ActiveStreamConflict:
            await db.execute("ROLLBACK")
            raise
        except Exception:
            await db.execute("ROLLBACK")
            raise


async def update_message_content(
    message_id: str,
    content: str,
    metadata: dict | None,
    status: str,
    error: str | None = None,
    tool_blocks: list[dict] | None = None,
) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE messages SET content=?, metadata=?, status=?, error=?, tool_blocks=? WHERE id=?",
            (
                content,
                json.dumps(metadata) if metadata is not None else None,
                status,
                error,
                json.dumps(tool_blocks) if tool_blocks is not None else None,
                message_id,
            ),
        )
        await db.commit()


async def set_active_stream(conversation_id: str, message_id: str | None) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE conversations SET active_stream_message_id=? WHERE id=?",
            (message_id, conversation_id),
        )
        await db.commit()


async def assert_message_in_list(message_id: str, list_id: str) -> bool:
    """Return True if message_id belongs to a conversation scoped to list_id."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT 1 FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.id=? AND c.list_id=?
            """,
            (message_id, list_id),
        )
        return await cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# FTS5 helpers
# ---------------------------------------------------------------------------


async def clear_fts_for_list(list_id: str) -> None:
    """Delete all FTS rows whose source_id belongs to sources in the given list."""
    async with get_db() as db:
        await db.execute(
            "DELETE FROM chunks_fts WHERE source_id IN (SELECT id FROM sources WHERE list_id = ?)",
            (list_id,),
        )
        await db.commit()


_CJK = re.compile(r"[一-鿿㐀-䶿]")  # BMP + Ext A; Ext B+ (U+20000+) and kana not covered — adequate for zh

_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")


def _cjk_runs(text: str):
    """Yield (is_cjk: bool, segment: str) pairs, splitting on CJK/non-CJK boundaries."""
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            yield (False, text[pos : m.start()])
        yield (True, m.group(0))
        pos = m.end()
    if pos < len(text):
        yield (False, text[pos:])


def _cjk_bigram_tokens(text: str) -> list[str]:
    """Produce unigrams + overlapping bigrams for each CJK run; split non-CJK on whitespace.

    Unigrams ensure single-char CJK words (e.g. 死, 岁, 杀) are searchable.
    Bigrams add adjacency signal so compound words rank above scattered co-occurrence.
    Length-1 CJK runs emit just the unigram. Non-CJK runs are split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if is_cjk:
            out += list(seg)  # unigrams
            out += [seg[i : i + 2] for i in range(len(seg) - 1)]  # bigrams
        else:
            out += seg.split()
    return out


def _collapse_cjk_whitespace(text: str) -> str:
    """Remove whitespace between adjacent CJK characters.

    chunk.py joins transcript segments with spaces; collapsing CJK-CJK gaps
    lets bigrams span them. Shared by the char and pinyin index tokenizers so
    both span segment joins identically.
    """
    return re.sub(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])", "", text)


def _tokenize_cjk(text: str) -> str:
    """Tokenize text for FTS5: unigrams + overlapping bigrams per CJK run.

    Whitespace between adjacent CJK characters is collapsed so character
    bigrams aren't split across it. Non-CJK text is split on whitespace.
    """
    return " ".join(_cjk_bigram_tokens(_collapse_cjk_whitespace(text)))


def _cjk_query_tokens(text: str) -> list[str]:
    """Produce FTS5 query tokens for a single whitespace-delimited word.

    Multi-char CJK runs emit overlapping bigrams only — no unigrams, to
    avoid AND-term explosion. Single-char CJK runs emit their unigram so
    high-IDF single-character words survive. Non-CJK split on whitespace.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if is_cjk:
            if len(seg) == 1:
                out.append(seg)
            else:
                out += [seg[i : i + 2] for i in range(len(seg) - 1)]
        else:
            out += seg.split()
    return out


def _pinyin_tokens(text: str) -> list[str]:
    """Toneless-pinyin syllable bigrams for each CJK run.

    Per CJK run: toneless syllables → overlapping bigrams (no unigrams).
    Single-char CJK runs produce nothing — pinyin unigrams are catastrophically
    noisy (~400 toneless syllables). Non-CJK runs are skipped entirely.
    """
    out: list[str] = []
    for is_cjk, seg in _cjk_runs(text):
        if not is_cjk:
            continue
        syllables = lazy_pinyin(seg, style=Style.NORMAL)
        if len(syllables) >= 2:
            out += [syllables[i] + syllables[i + 1] for i in range(len(syllables) - 1)]
    return out


def _pinyin_index_tokens(text: str) -> str:
    """Toneless-pinyin syllable bigrams, space-joined for FTS5 index column.

    Collapses CJK-CJK whitespace first (same as _tokenize_cjk) so pinyin
    bigrams span segment-join gaps — without it the pinyin arm would miss
    homophones straddling a segment boundary that the char arm catches.
    """
    return " ".join(_pinyin_tokens(_collapse_cjk_whitespace(text)))


def _fts_quote_token(t: str) -> str:
    """Double-quote and escape a single FTS5 token (disables operator parsing)."""
    return '"' + t.replace('"', '""') + '"'


def _escape_fts_query(query_text: str) -> str:
    """Escape user input for safe FTS5 MATCH evaluation.

    Returns a column-scoped expression targeting the content and/or pinyin
    FTS5 columns so the BM25 arm survives homophone ASR errors (e.g. query
    '苹果' matches indexed '平果' via shared toneless-pinyin bigrams):

        content : ("a" OR "b") OR pinyin : ("x" OR "y")

    Tokens within an arm are OR-joined, not AND-joined: a multi-syllable term or
    a natural-language question ('苹果是什么') is one CJK run whose overlapping
    bigrams must not all be required to co-occur — under AND a single divergent
    syllable (or an interrogative like '是谁') zeroes the arm, so recall collapsed
    for anything past a two-character word (#391). OR lets BM25 rank by matched
    bigrams; high-IDF entity bigrams dominate and stop-word bigrams sit in the
    truncated tail. Each arm parenthesizes its token list so the column filter
    binds to the whole group — FTS5's `col :` prefix otherwise scopes only the
    first phrase, leaving later tokens to probe every column. With the parens,
    char tokens never probe pinyin and pinyin tokens never probe content. When
    the pinyin arm produces no tokens (English, single-char CJK, no CJK at all)
    the OR is omitted and only the content arm is returned.
    """
    words = query_text.split()
    char_tokens = [tok for word in words for tok in _cjk_query_tokens(word)]
    py_tokens = [tok for word in words for tok in _pinyin_tokens(word)]

    if not char_tokens and not py_tokens:
        return ""

    char_arm = ""
    if char_tokens:
        char_arm = "content : (" + " OR ".join(_fts_quote_token(t) for t in char_tokens) + ")"

    py_arm = ""
    if py_tokens:
        py_arm = "pinyin : (" + " OR ".join(_fts_quote_token(t) for t in py_tokens) + ")"

    if char_arm and py_arm:
        return f"{char_arm} OR {py_arm}"
    return char_arm or py_arm


async def query_fts_rows(
    query_text: str,
    source_ids: list[str],
    top_k: int = 30,
) -> list[aiosqlite.Row]:
    """Run FTS5 MATCH query filtered by source_ids, return rows ranked by BM25.

    Returns an empty list if the query is empty or FTS5 raises a syntax error.
    """
    if not source_ids:
        return []
    match_query = _escape_fts_query(query_text)
    if not match_query:
        return []
    placeholders = _in_placeholders(source_ids)
    async with get_db() as db:
        try:
            cursor = await db.execute(
                f"SELECT content, source_id, video_title, timestamp_start, timestamp_end, rank, chunk_id, "
                f"seg_start, seg_end "
                f"FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? AND source_id IN ({placeholders}) "
                f"ORDER BY rank "
                f"LIMIT ?",
                [match_query, *source_ids, top_k],
            )
            return await cursor.fetchall()
        except aiosqlite.OperationalError as exc:
            logger.warning("FTS5 MATCH query failed (%s); returning empty results", exc)
            return []
