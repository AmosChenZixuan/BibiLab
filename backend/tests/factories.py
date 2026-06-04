"""DB setup factories for tests.

These factories isolate test setup from production `db.py` public-API shape.
A column-shape change in `sources` / `conversations` / `messages` only needs
to touch this file (or `_DEFAULTS`), not 178 call sites scattered across
9 test files.

Design:
- `SourceFactory.build` calls the production `write_source_with_segments(segments=[])`
  with sensible defaults from `_DEFAULTS`. Tests override per-call as needed.
  Decoupling is signature-only: the SQL (ON CONFLICT clause, COALESCE facet
  semantics, DELETE-before-INSERT ordering, thumbnail auto-assign) lives in
  one place — `_exec_write_source` in `db.py`. A schema change there
  automatically reaches the factory.
- `ConversationFactory.build` and `MessageFactory.build` use raw SQL because
  there is no production public function for those inserts (production
  uses `get_or_create_conversation` / `create_user_and_assistant_atomic`,
  which are different shapes). The factory's INSERTs are minimal column
  sets; a column addition only touches this file.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from bibilab.db import get_db, write_source_with_segments


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceFactory:
    """Build a `sources` row. Default to all required fields with sensible
    placeholders; override per-call as needed. Returns the new source_id."""

    _DEFAULTS: dict[str, Any] = {
        "video_id": "BV1xxx",
        "platform": "bilibili",
        "title": "Test Video",
        "summary": "",
        "keywords": [],
        "cover_url": None,
        "source_url": "https://www.bilibili.com/video/BV1xxx",
        "duration_seconds": 0,
        "uploader": "TestUploader",
        "language": None,
        "whisper_model": "large-v3",
        "ai_model": "gpt-4o",
        "vision_enabled": False,
        "settings_snapshot": {},
        "series_name": None,
        "sequence_number": None,
        "season_number": None,
    }

    @classmethod
    async def build(cls, list_id: str, **overrides: Any) -> str:
        """Insert a sources row via the production write path.

        A column addition in `sources` only requires updating `_DEFAULTS`;
        the INSERT/ON CONFLICT/COALESCE/thumbnail side effect live in
        `db._exec_write_source` and apply here automatically.
        """
        source_id = overrides.pop("source_id", None) or str(uuid.uuid4())
        fields = {**cls._DEFAULTS, **overrides, "list_id": list_id, "source_id": source_id}
        await write_source_with_segments(segments=[], **fields)
        return source_id


class ConversationFactory:
    """Build a `conversations` row. Returns the new conversation_id."""

    @classmethod
    async def build(cls, list_id: str) -> str:
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


class MessageFactory:
    """Build a `messages` row. Returns the row (caller can read msg['id'] etc.)."""

    _DEFAULTS: dict[str, Any] = {
        "role": "user",
        "content": "",
        "metadata": None,
    }

    @classmethod
    async def build(cls, conversation_id: str, **overrides: Any) -> aiosqlite.Row:
        fields = {**cls._DEFAULTS, **overrides}
        message_id = str(uuid.uuid4())
        now = _now()
        metadata_json = json.dumps(fields["metadata"]) if fields["metadata"] is not None else None
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO messages
                    (id, conversation_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    fields["role"],
                    fields["content"],
                    metadata_json,
                    now,
                ),
            )
            await db.commit()
            cursor = await db.execute("SELECT * FROM messages WHERE id=?", (message_id,))
            return await cursor.fetchone()
