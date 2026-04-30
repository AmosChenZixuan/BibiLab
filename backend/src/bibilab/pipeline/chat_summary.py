"""Conversation summary compression — roll up old messages into a summary."""

import asyncio
import logging

from bibilab.config import BibilabConfig
from bibilab.db import (
    compress_conversation,
    get_conversation,
    get_message_count,
    get_messages_beyond_window,
)
from bibilab.pipeline._shared import _call_llm

logger = logging.getLogger(__name__)

COMPRESSION_THRESHOLD = 30
SLIDING_WINDOW_SIZE = 20
# Target length the model should aim for in the summary text itself.
SUMMARY_TARGET_TOKENS = 500
# API budget — must accommodate thinking tokens + the ~500-token output for reasoning models.
SUMMARY_MAX_TOKENS = 8192


async def maybe_compress_conversation(
    conversation_id: str,
    cfg: BibilabConfig,
) -> None:
    count = await get_message_count(conversation_id)
    if count <= COMPRESSION_THRESHOLD:
        return

    conv = await get_conversation(conversation_id)
    if conv is None:
        return

    existing_summary = conv["summary"]

    to_compress = await get_messages_beyond_window(conversation_id, SLIDING_WINDOW_SIZE)
    if not to_compress:
        return

    messages_text = "\n".join(f"{row['role']}: {row['content']}" for row in to_compress)

    if existing_summary:
        compression_prompt = (
            f"Existing summary of the conversation so far:\n"
            f"{existing_summary}\n\n"
            f"New messages to integrate into the summary:\n"
            f"{messages_text}\n\n"
            "Compress the above messages and merge them with the existing summary. "
            f"Produce a concise summary (ideally under {SUMMARY_TARGET_TOKENS} tokens) that preserves key facts, "
            "decisions, user preferences, and topics discussed. "
            "Return only the updated summary text, no preamble or explanation."
        )
    else:
        compression_prompt = (
            f"Compress the following conversation history into a concise summary. "
            f"Preserve key facts, decisions, user preferences, and topics discussed. "
            f"Target length: under {SUMMARY_TARGET_TOKENS} tokens. "
            f"Return only the summary text, no preamble or explanation.\n\n"
            f"{messages_text}"
        )

    try:
        new_summary = await asyncio.to_thread(
            _call_llm,
            prompt=compression_prompt,
            cfg=cfg.ai,
            llm_max_tokens=SUMMARY_MAX_TOKENS,
        )
    except Exception:
        logger.exception("Failed to compress conversation %s", conversation_id)
        return

    ids_to_delete = [row["id"] for row in to_compress]
    await compress_conversation(conversation_id, new_summary, ids_to_delete)
