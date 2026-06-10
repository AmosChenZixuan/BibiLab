"""Messages table CRUD + message-status constants.

The turn-creation and turn-terminal transactions live in
``bibilab.pipeline.chat_runs`` (they are domain rules over the messages +
conversations tables, not pure SQL). The CRUD below is the SQL surface.
"""

from __future__ import annotations

import aiosqlite

from bibilab.db.connection import _now, get_db
from bibilab.db.sources import _in_placeholders

# A turn is visible to LLM replay and compaction iff both rows have
# status='done'. The two transient row states used during a turn are
# IN_FLIGHT_USER_STATUS (user awaiting its assistant) and
# IN_FLIGHT_ASST_STATUS (assistant actively generating). The startup sweep
# considers both in-flight states and flips them to 'failed' on startup.
# Replay/compaction queries filter on VISIBLE only. Terminal 'failed' /
# 'cancelled' are NOT in-flight — they are terminal-but-invisible to the
# LLM (replay filter) and to the user (separate UI path) but not subject
# to the startup sweep.
VISIBLE_MESSAGE_STATUS: str = "done"
IN_FLIGHT_USER_STATUS: str = "pending"
IN_FLIGHT_ASST_STATUS: str = "streaming"
IN_FLIGHT_MESSAGE_STATUSES: tuple[str, ...] = (
    IN_FLIGHT_ASST_STATUS,
    IN_FLIGHT_USER_STATUS,
)


async def get_recent_messages(
    conversation_id: str,
    limit: int,
    before_id: str | None = None,
) -> list[aiosqlite.Row]:
    # Unfiltered by status — both the UI history endpoint and the LLM
    # history snapshot read from this. The UI must show cancelled/failed
    # rows (已停止/重试); the LLM snapshot site filters to VISIBLE inline.
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
            "SELECT COUNT(*) FROM messages WHERE conversation_id=? AND status=?",
            (conversation_id, VISIBLE_MESSAGE_STATUS),
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
                WHERE conversation_id=? AND status=?
            )
            WHERE _rn > ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id, VISIBLE_MESSAGE_STATUS, window_size),
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
