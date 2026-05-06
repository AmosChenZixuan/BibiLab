"""Tests for conversation summary compression."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_no_compression_below_threshold(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        get_conversation,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    for i in range(COMPRESSION_THRESHOLD - 1):
        await create_message(conv_id, "user", f"Message {i}", None)

    assert await get_message_count(conv_id) == COMPRESSION_THRESHOLD - 1

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    from unittest.mock import patch

    from bibilab.pipeline.chat_summary import _call_llm as real_call_llm

    called = False

    def fake_call_llm(*args, **kwargs):
        nonlocal called
        called = True
        return real_call_llm(*args, **kwargs)

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    assert not called, "Compression should not have been triggered"
    conv = await get_conversation(conv_id)
    assert conv["summary"] is None


@pytest.mark.asyncio
async def test_compression_triggers_above_threshold(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        get_conversation,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    for i in range(COMPRESSION_THRESHOLD + 5):
        await create_message(conv_id, "user", f"Message {i}", None)
        await asyncio.sleep(0.001)

    assert await get_message_count(conv_id) == COMPRESSION_THRESHOLD + 5

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    called_with = None

    def fake_call_llm(*args, **kwargs):
        nonlocal called_with
        called_with = (args, kwargs)
        return "Test summary."

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    assert called_with is not None, "Compression should have been triggered"
    conv = await get_conversation(conv_id)
    assert conv["summary"] == "Test summary."
    remaining = await get_message_count(conv_id)
    assert remaining == SLIDING_WINDOW_SIZE


@pytest.mark.asyncio
async def test_existing_summary_included_in_prompt(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        update_conversation_summary,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    await update_conversation_summary(conv_id, "User loves Python tutorials.")

    for i in range(COMPRESSION_THRESHOLD + 3):
        await create_message(conv_id, "user", f"Message {i}", None)
        await asyncio.sleep(0.001)

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    captured_prompt = None

    def fake_call_llm(prompt, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "Updated summary."

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    assert captured_prompt is not None
    assert "User loves Python tutorials." in captured_prompt
    assert "Message 0" in captured_prompt
    assert "PRESERVE ALL [title @ Ts-Ts] citations" in captured_prompt


@pytest.mark.asyncio
async def test_compression_deletes_old_messages(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    for i in range(COMPRESSION_THRESHOLD + 5):
        await create_message(conv_id, "user", f"Message {i}", None)
        await asyncio.sleep(0.001)

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    def fake_call_llm(*args, **kwargs):
        return "Summary."

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    remaining = await get_message_count(conv_id)
    assert remaining == SLIDING_WINDOW_SIZE


@pytest.mark.asyncio
async def test_no_compression_when_no_messages_to_compress(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
    )
    from bibilab.pipeline.chat_summary import SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    for i in range(SLIDING_WINDOW_SIZE + 5):
        await create_message(conv_id, "user", f"Message {i}", None)
        await asyncio.sleep(0.001)

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    from bibilab.pipeline.chat_summary import _call_llm as real_call_llm

    called = False

    def fake_call_llm(*args, **kwargs):
        nonlocal called
        called = True
        return real_call_llm(*args, **kwargs)

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    assert not called, "Should not compress when all messages are within window"


@pytest.mark.asyncio
async def test_nonexistent_conversation_noops(tmp_bibilab_home):
    from bibilab.db import bootstrap_db
    from bibilab.pipeline.chat_summary import maybe_compress_conversation

    await bootstrap_db()

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    from bibilab.pipeline.chat_summary import _call_llm as real_call_llm

    called = False

    def fake_call_llm(*args, **kwargs):
        nonlocal called
        called = True
        return real_call_llm(*args, **kwargs)

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation("nonexistent-conv-id", cfg)

    assert not called


@pytest.mark.asyncio
async def test_compression_prompt_preserves_citations(tmp_bibilab_home):
    """Compression prompt must include instruction to preserve [title @ Ts-Ts] citations."""
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    for i in range(COMPRESSION_THRESHOLD + 3):
        await create_message(conv_id, "user", f"Message {i}", None)
        await asyncio.sleep(0.001)

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    from unittest.mock import patch

    captured_prompt = None

    def fake_call_llm(prompt, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "Summary with citation preserved."

    with patch("bibilab.pipeline.chat_summary._call_llm", fake_call_llm):
        await maybe_compress_conversation(conv_id, cfg)

    assert captured_prompt is not None
    assert "PRESERVE ALL [title @ Ts-Ts] citations" in captured_prompt


@pytest.mark.asyncio
async def test_get_message_count_empty(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        get_message_count,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    assert await get_message_count(conv_id) == 0


@pytest.mark.asyncio
async def test_get_message_count_after_creates(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_conversation,
        create_list,
        create_message,
        get_message_count,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await create_conversation("list-1")

    await create_message(conv_id, "user", "Hello", None)
    await create_message(conv_id, "assistant", "Hi", None)

    assert await get_message_count(conv_id) == 2


@pytest.mark.asyncio
async def test_get_messages_beyond_window_returns_older_messages(tmp_bibilab_home):
    from datetime import datetime, timedelta

    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()

    conv_id = "test-conv-1"
    base = datetime(2026, 1, 1, 0, 0, 0)
    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("test-list-1", "Test List", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO conversations (id, list_id, summary, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)",
            (conv_id, "test-list-1", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )

        for i in range(12):
            ts = (base + timedelta(seconds=i)).isoformat()
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, "
                "metadata, created_at) VALUES (?, ?, ?, ?, NULL, ?)",
                (f"msg-{i:02d}", conv_id, "user", f"Message {i}", ts),
            )
        await db.commit()

    from bibilab.db import get_messages_beyond_window

    window_size = 5
    beyond = await get_messages_beyond_window(conv_id, window_size)
    beyond_ids = {r["id"] for r in beyond}

    assert len(beyond) == 12 - window_size
    assert "msg-00" in beyond_ids
    assert "msg-06" in beyond_ids
    assert "msg-07" not in beyond_ids
    assert "msg-11" not in beyond_ids
