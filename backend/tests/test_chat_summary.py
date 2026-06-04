"""Tests for conversation summary compression."""

import pytest

from tests.factories import ConversationFactory, MessageFactory


@pytest.mark.asyncio
async def test_no_compression_below_threshold(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_conversation,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    for i in range(COMPRESSION_THRESHOLD - 1):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    assert await get_message_count(conv_id) == COMPRESSION_THRESHOLD - 1

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 0, "Compression should not have been triggered"
    conv = await get_conversation(conv_id)
    assert conv["summary"] is None


@pytest.mark.asyncio
async def test_compression_triggers_above_threshold(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_conversation,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    for i in range(COMPRESSION_THRESHOLD + 5):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    assert await get_message_count(conv_id) == COMPRESSION_THRESHOLD + 5

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    mock_call_llm.return_value = "Test summary."
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 1, "Compression should have been triggered"
    conv = await get_conversation(conv_id)
    assert conv["summary"] == "Test summary."
    remaining = await get_message_count(conv_id)
    assert remaining == SLIDING_WINDOW_SIZE


@pytest.mark.asyncio
async def test_existing_summary_included_in_prompt(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import (
        bootstrap_db,
        compress_conversation,
        create_list,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    # Seed an existing summary via the production write path (no messages to delete).
    await compress_conversation(conv_id, "User loves Python tutorials.", [])

    for i in range(COMPRESSION_THRESHOLD + 3):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    def fake_call_llm(prompt, *args, **kwargs):
        return "Updated summary."

    mock_call_llm.side_effect = fake_call_llm
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 1
    captured_prompt = mock_call_llm.call_args.kwargs["prompt"]
    assert "User loves Python tutorials." in captured_prompt
    assert "Message 0" in captured_prompt
    # Legacy [title @ Ts-Ts] citation preservation was removed per #241 spec:
    # summaries are plain prose without legacy token preservation.
    assert "PRESERVE ALL [title @ Ts-Ts]" not in captured_prompt


@pytest.mark.asyncio
async def test_compression_deletes_old_messages(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_message_count,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    for i in range(COMPRESSION_THRESHOLD + 5):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    mock_call_llm.return_value = "Summary."
    await maybe_compress_conversation(conv_id, cfg)

    remaining = await get_message_count(conv_id)
    assert remaining == SLIDING_WINDOW_SIZE


@pytest.mark.asyncio
async def test_no_compression_when_no_messages_to_compress(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import (
        bootstrap_db,
        create_list,
    )
    from bibilab.pipeline.chat_summary import SLIDING_WINDOW_SIZE, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    for i in range(SLIDING_WINDOW_SIZE + 5):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 0, "Should not compress when all messages are within window"


@pytest.mark.asyncio
async def test_nonexistent_conversation_noops(tmp_bibilab_home, mock_call_llm):
    from bibilab.db import bootstrap_db
    from bibilab.pipeline.chat_summary import maybe_compress_conversation

    await bootstrap_db()

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    await maybe_compress_conversation("nonexistent-conv-id", cfg)

    assert mock_call_llm.call_count == 0


@pytest.mark.asyncio
async def test_compression_prompt_no_legacy_citation_preservation(tmp_bibilab_home, mock_call_llm):
    """Compression prompt does not preserve legacy [title @ Ts-Ts] citations per #241 spec."""
    from bibilab.db import (
        bootstrap_db,
        create_list,
    )
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    for i in range(COMPRESSION_THRESHOLD + 3):
        await MessageFactory.build(
            conv_id,
            content=f"Message {i}",
        )

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()

    def fake_call_llm(prompt, *args, **kwargs):
        return "Summary with citation preserved."

    mock_call_llm.side_effect = fake_call_llm
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 1
    captured_prompt = mock_call_llm.call_args.kwargs["prompt"]
    # Legacy [title @ Ts-Ts] citation preservation was removed per #241 spec:
    # summaries are plain prose without legacy token preservation.
    assert "PRESERVE ALL [title @ Ts-Ts]" not in captured_prompt


@pytest.mark.asyncio
async def test_get_message_count_empty(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_message_count,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    assert await get_message_count(conv_id) == 0


@pytest.mark.asyncio
async def test_get_message_count_after_creates(tmp_bibilab_home):
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_message_count,
    )

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    await MessageFactory.build(
        conv_id,
        content="Hello",
    )
    await MessageFactory.build(
        conv_id,
        role="assistant",
        content="Hi",
    )

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


# --- #403: aborted turns do not count toward the compression trigger ---


@pytest.mark.asyncio
async def test_aborted_messages_do_not_trigger_compression(tmp_bibilab_home, mock_call_llm):
    """Cancelled/failed rows must not push the count past the >30 threshold —
    otherwise we'd summarize blank assistant turns."""
    from bibilab.db import bootstrap_db, create_list, get_message_count
    from bibilab.pipeline.chat_summary import COMPRESSION_THRESHOLD, maybe_compress_conversation

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    conv_id = await ConversationFactory.build("list-1")

    # Seed THRESHOLD done messages plus a generous number of aborted rows.
    # Both rows of each aborted turn share the same terminal status (the #403
    # invariant — the user message flips to 'cancelled'/'failed' alongside the
    # assistant in run_chat_turn's finally block).
    for i in range(COMPRESSION_THRESHOLD):
        await MessageFactory.build(conv_id, content=f"msg-{i}", status="done")
    for _ in range(20):
        await MessageFactory.build(conv_id, role="user", content="u-aborted", status="cancelled")
        await MessageFactory.build(conv_id, role="assistant", content="", status="cancelled")

    # get_message_count should now report only the done rows.
    assert await get_message_count(conv_id) == COMPRESSION_THRESHOLD

    from bibilab.config import BibilabConfig

    cfg = BibilabConfig()
    await maybe_compress_conversation(conv_id, cfg)

    assert mock_call_llm.call_count == 0, "Compression must not fire when done-count is at the threshold"
