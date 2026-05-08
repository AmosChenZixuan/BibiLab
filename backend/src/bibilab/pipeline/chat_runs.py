"""In-memory registry for active chat producer tasks and their SSE buffers.

Decouples LLM runs from HTTP request lifetime — see
docs/specs/2026-05-07-resumable-chat-streams-design.md (local-only).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal

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
