"""In-memory registry for active chat producer tasks and their SSE buffers.

Decouples LLM runs from HTTP request lifetime — the producer task writes
events into a StreamBuffer that any consumer (POST handler, GET reattach)
can drain, so a disconnected or reattached HTTP request no longer cancels
the LLM turn in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal

from bibilab.db import IN_FLIGHT_ASST_STATUS, IN_FLIGHT_USER_STATUS, _now, get_db

logger = logging.getLogger(__name__)

STREAM_BUFFER_GRACE_SECONDS = 60

TerminalStatus = Literal["done", "failed", "cancelled"]


@dataclass
class StreamBuffer:
    message_id: str
    events: list[dict] = field(default_factory=list)
    subscribers: set[asyncio.Event] = field(default_factory=set)
    terminal: bool = False
    final_status: TerminalStatus | None = None

    def append(self, event: dict) -> None:
        self.events.append(event)
        for ev in self.subscribers:
            ev.set()

    def close(self, status: TerminalStatus) -> None:
        self.terminal = True
        self.final_status = status
        for ev in self.subscribers:
            ev.set()


async def stream_from_buffer(buf: StreamBuffer) -> AsyncGenerator[dict, None]:
    cursor = 0
    wake = asyncio.Event()
    buf.subscribers.add(wake)
    try:
        while True:
            while cursor < len(buf.events):
                yield buf.events[cursor]
                cursor += 1
            if buf.terminal:
                return
            wake.clear()
            await wake.wait()
    finally:
        buf.subscribers.discard(wake)


class ChatRunRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[StreamBuffer, asyncio.Task]] = {}
        self._bg_tasks: set[asyncio.Task] = set()

    def register(self, message_id: str, task: asyncio.Task) -> StreamBuffer:
        if message_id in self._entries:
            raise ValueError(f"message_id already registered: {message_id}")
        buf = StreamBuffer(message_id=message_id)
        self._entries[message_id] = (buf, task)
        return buf

    def get(self, message_id: str) -> StreamBuffer | None:
        entry = self._entries.get(message_id)
        return entry[0] if entry else None

    def cancel(self, message_id: str) -> bool:
        entry = self._entries.get(message_id)
        if entry is None:
            return False
        _, task = entry
        if task.done():
            logger.info("cancel on already-completed task message_id=%s", message_id)
            return False
        task.cancel()
        return True

    def evict(self, message_id: str) -> None:
        self._entries.pop(message_id, None)

    def all_message_ids(self) -> list[str]:
        return list(self._entries.keys())

    def all_tasks(self) -> list[tuple[str, asyncio.Task]]:
        return [(mid, task) for mid, (_, task) in self._entries.items()]

    def track_background(self, task: asyncio.Task) -> None:
        self._bg_tasks.add(task)
        task.add_done_callback(lambda t: self._bg_tasks.discard(t))

    def all_background_tasks(self) -> list[asyncio.Task]:
        return list(self._bg_tasks)


_registry = ChatRunRegistry()


def get_chat_run_registry() -> ChatRunRegistry:
    return _registry


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
                "VALUES (?, ?, 'user', ?, NULL, ?, ?)",
                (user_msg_id, conversation_id, user_text, now, IN_FLIGHT_USER_STATUS),
            )
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at, status) "
                "VALUES (?, ?, 'assistant', '', NULL, ?, ?)",
                (assistant_msg_id, conversation_id, now, IN_FLIGHT_ASST_STATUS),
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


async def update_turn_terminal(
    *,
    conversation_id: str,
    user_msg_id: str,
    asst_msg_id: str,
    asst_content: str,
    asst_metadata: dict | None,
    asst_tool_blocks: list[dict] | None,
    status: str,
    error: str | None,
) -> None:
    """Atomically flip a turn to its terminal status and clear the
    conversation's active-stream pointer.

    All three writes — user row, assistant row, conversations.active_stream_message_id
    — commit in one transaction so a process kill between them cannot
    strand an orphan or leave a wedged active-stream pointer that 409s
    future POSTs. The user row's content/metadata/tool_blocks are unchanged
    from insert time, so the user UPDATE is narrowed to status + error
    only. The conversation's user row also gets a NULL error (the
    assistant failed, not the user). Both message UPDATEs assert rowcount
    to catch stale ids (programmer error).
    """
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE messages SET status=?, error=NULL WHERE id=?",
            (status, user_msg_id),
        )
        if cur.rowcount == 0:
            raise RuntimeError(f"update_turn_terminal: user_msg_id={user_msg_id!r} not found")
        cur = await db.execute(
            "UPDATE messages SET content=?, metadata=?, status=?, error=?, tool_blocks=? WHERE id=?",
            (
                asst_content,
                json.dumps(asst_metadata) if asst_metadata is not None else None,
                status,
                error,
                json.dumps(asst_tool_blocks) if asst_tool_blocks is not None else None,
                asst_msg_id,
            ),
        )
        if cur.rowcount == 0:
            raise RuntimeError(f"update_turn_terminal: asst_msg_id={asst_msg_id!r} not found")
        # Clear the active-stream pointer in the same transaction so a
        # crash here cannot wedge a future POST with HTTP 409.
        await db.execute(
            "UPDATE conversations SET active_stream_message_id=NULL WHERE id=?",
            (conversation_id,),
        )
        await db.commit()
