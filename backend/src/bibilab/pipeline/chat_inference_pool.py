"""Dedicated ThreadPoolExecutor for chat-path inference (rerank + ChromaDB query).

Isolates chat inference from the shared default executor used by
ingestion pipeline stages (transcribe/embed/digest).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

_chat_pool: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def get_chat_pool() -> ThreadPoolExecutor:
    """Return the shared chat inference thread pool (lazy init, thread-safe)."""
    global _chat_pool
    if _chat_pool is None:
        with _lock:
            if _chat_pool is None:
                _chat_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat-inf")
    return _chat_pool
