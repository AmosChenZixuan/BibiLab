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
from bibilab.pipeline.digest import SectionDigest
from bibilab.pipeline.section import Section
from bibilab.pipeline.transcribe import WhisperSegment


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _insert_message(
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    metadata_json: str | None,
    now: str,
    status: str,
) -> aiosqlite.Row:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO messages
                (id, conversation_id, role, content, metadata, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, metadata_json, now, status),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM messages WHERE id=?", (message_id,))
        return await cursor.fetchone()


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
    async def build(cls, list_id: str, *, active_stream_message_id: str | None = None) -> str:
        conversation_id = str(uuid.uuid4())
        now = _now()
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO conversations (id, list_id, summary, created_at, updated_at, active_stream_message_id)
                VALUES (?, ?, NULL, ?, ?, ?)
                """,
                (conversation_id, list_id, now, now, active_stream_message_id),
            )
            await db.commit()
        return conversation_id


class MessageFactory:
    """Build a `messages` row. Returns the row (caller can read msg['id'] etc.).

    `message_id` defaults to a fresh uuid; pass an explicit id to assert on
    a specific value (e.g. to pair a user/asst row in turn-transition tests).
    """

    _DEFAULTS: dict[str, Any] = {
        "role": "user",
        "content": "",
        "metadata": None,
        "status": "done",  # matches schema DEFAULT
    }

    @classmethod
    async def build(cls, conversation_id: str, **overrides: Any) -> aiosqlite.Row:
        fields = {**cls._DEFAULTS, **overrides}
        message_id = fields.pop("message_id", None) or str(uuid.uuid4())
        metadata_json = json.dumps(fields["metadata"]) if fields["metadata"] is not None else None
        return await _insert_message(
            message_id,
            conversation_id,
            fields["role"],
            fields["content"],
            metadata_json,
            _now(),
            fields["status"],
        )


class SectionedSourceFactory:
    """Factory for sources with sections and segment rows pre-populated.

    Returns (source_id, segments, sections, section_digests). Defaults: 30
    segments of 1-second `s{i}.` text → 3 sections of 10 segments each.
    Pass `n_segments` to change segment count; `n_sections` overrides the
    derived default (3). `section_digests` defaults to one placeholder
    `SectionDigest(summary=f"S{i} section.", keywords=[f"k{i}"])` per
    section. Pass any of these to override.
    """

    @staticmethod
    async def build(
        list_id: str,
        *,
        n_segments: int = 30,
        n_sections: int | None = None,
        section_digests: list[SectionDigest] | None = None,
        **source_overrides: Any,
    ) -> tuple[str, list[WhisperSegment], list[Section], list[SectionDigest]]:
        source_id = source_overrides.pop("source_id", None) or str(uuid.uuid4())
        segments = [
            WhisperSegment(start=float(i), end=float(i + 1), text=f"s{i}.", speaker=None) for i in range(n_segments)
        ]
        # Derive n_sections from the segment count if not given.
        if n_sections is None:
            n_sections = max(1, n_segments // 10)
        segs_per_section = max(1, n_segments // n_sections)
        sections: list[Section] = []
        for s_idx in range(n_sections):
            seg_start = s_idx * segs_per_section
            seg_end = min(seg_start + segs_per_section - 1, n_segments - 1)
            if seg_start > n_segments - 1:
                break
            sections.append(
                Section(
                    seg_start=seg_start,
                    seg_end=seg_end,
                    token_count=100,
                    timestamp_start=float(seg_start),
                    timestamp_end=float(seg_end + 1),
                )
            )
        if section_digests is None:
            section_digests = [
                SectionDigest(summary=f"S{i} section.", keywords=[f"k{i}"]) for i in range(len(sections))
            ]
        # Reuse SourceFactory's defaults; the source_id must match.
        merged: dict[str, Any] = {
            **SourceFactory._DEFAULTS,
            **source_overrides,
            "list_id": list_id,
            "source_id": source_id,
        }
        await write_source_with_segments(
            segments=segments,
            sections=sections,
            section_digests=section_digests,
            **merged,
        )
        return source_id, segments, sections, section_digests
