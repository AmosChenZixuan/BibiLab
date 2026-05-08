import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from bibilab.config import AIConfig, BibilabConfig, get_config
from bibilab.db import (
    ActiveStreamConflict,
    assert_message_in_list,
    create_user_and_assistant_atomic,
    delete_conversation,
    get_conversation_by_list,
    get_list,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_for_list,
    set_active_stream,
    update_message_content,
)
from bibilab.db import (
    get_conversation as get_conv_row,
)
from bibilab.models.chat import (
    ChatRequest,
    ConversationResponse,
    GetConversationResponse,
    MessageResponse,
)
from bibilab.pipeline._shared import StreamEvent, ToolCall, ToolDefinition, stream_llm
from bibilab.pipeline.chat_runs import (
    STREAM_BUFFER_GRACE_SECONDS,
    ChatRunRegistry,
    StreamBuffer,
    TerminalStatus,
    get_chat_run_registry,
    stream_from_buffer,
)
from bibilab.pipeline.chat_summary import maybe_compress_conversation
from bibilab.pipeline.chat_tools import (
    GENERATE_REPORT_TOOL,
    QUERY_LIST_METADATA_TOOL,
    RETRIEVE_TOOL,
    CitationRegistryEntry,
    execute_tool,
)
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta

logger = logging.getLogger(__name__)

router = APIRouter()

# SSE event types — used both as stream-internal event discriminators (stream_llm yield)
# and as the 'type' field in the SSE 'data:' JSON payload sent to the client.
SSE_EVENT_DELTA = "delta"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"
SSE_EVENT_TOOL_RESULT = "tool_result"
SSE_EVENT_TOOL_CALL_START = "tool_call_start"
SSE_EVENT_CITATION = "citation"
SSE_EVENT_CANCELLED = "cancelled"

# Sized for thinking-capable models with potentially long chat responses + tool turns.
CHAT_MAX_TOKENS = 16384

LOOPBACK_TOOLS = {"retrieve", "query_list_metadata"}
MAX_TOOL_ITERATIONS = 3

_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


def _flush_pending_text(content_blocks: list[dict], text: str) -> None:
    """Split text on paragraph boundaries (\n\n+), emit text + paragraph_break blocks."""
    if not text:
        return
    parts = _PARAGRAPH_SPLIT.split(text)
    for j, part in enumerate(parts):
        if part:
            content_blocks.append({"type": "text", "text": part})
        if j < len(parts) - 1:
            content_blocks.append({"type": "paragraph_break"})


@router.get("/lists/{list_id}/conversation")
async def get_conversation(
    list_id: str,
    before: str | None = None,
    limit: int = 50,
) -> GetConversationResponse:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    conversation_row = await get_conversation_by_list(list_id)
    if conversation_row is None:
        return GetConversationResponse(conversation=None, messages=[])

    messages_rows = await get_recent_messages(
        conversation_row["id"],
        limit=limit,
        before_id=before,
    )

    return GetConversationResponse(
        conversation=ConversationResponse.from_row(dict(conversation_row)),
        messages=[MessageResponse.from_row(dict(r)) for r in messages_rows],
    )


@router.delete("/lists/{list_id}/conversation", status_code=204)
async def delete_conversation_endpoint(list_id: str) -> None:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    conversation_row = await get_conversation_by_list(list_id)
    if conversation_row is not None:
        await delete_conversation(conversation_row["id"])


def _client_tool_result(result: dict) -> dict:
    """Strip internal fields (_-prefixed keys) before sending tool results to the client.

    Tool implementations may attach private metadata via _-prefixed keys (e.g. _chunks).
    These are never exposed over SSE. If you add a new tool whose result includes fields
    the client needs, do NOT prefix them with ``_``.
    """
    return {k: v for k, v in result.items() if not k.startswith("_")}


GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly based on the provided source material. "
    "CRITICAL RULES:\n"
    "0. If the user's question requires looking up video content (facts, comparisons, "
    "summaries), you MUST call the retrieve tool BEFORE answering. "
    "For questions about counts, durations, or languages of the sources themselves, "
    "call query_list_metadata instead. "
    "Do not answer from memory — always call the appropriate tool first for content "
    "or metadata questions. This does NOT apply when the user asks you to generate a "
    "report/artifact — the generate_report tool handles its own retrieval.\n"
    "1. ONLY use information from the source material provided below. Never use your own knowledge.\n"
    '2. If the excerpts do not contain the answer, say "The provided sources do not cover this topic."\n'
    "3. When the user identifies errors in your previous answer, audit ALL items in your prior response, "
    "not only the ones explicitly flagged — the user may have pointed out a sample. Re-verify each prior "
    "claim against the retrieved sources before re-asserting it.\n"
    "4. Quote or closely paraphrase the source material — do not reinterpret, editorialize, or add external context.\n"
    "5. Cite using exactly [N], where N is the source number from the retrieve result. "
    "Do not cite sources you did not retrieve. "
    "When citing content from a long source, mention the relevant timestamp inline in your prose "
    "(e.g. 'around the 2:00 mark [1]...' or 'between 1:24:30 and 1:25:10 [1]...'). "
    "Use natural phrasing, not a structured format. Skip timestamps for short sources or thematic citations.\n"
    "6. Use the generate_report tool when the user asks for summaries, study guides, blog posts, or custom reports.\n"
    "7. Do not ask follow-up questions, suggest next steps, or offer unsolicited advice.\n"
    "8. Be concise and direct. Answer in 1-3 sentences when possible."
)


async def stream_with_tools(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition],
    execute_tool_fn,
    system: str | None = None,
    llm_max_tokens: int = CHAT_MAX_TOKENS,
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    messages = list(messages)
    iteration = 0
    parse_buffer = ""
    citation_emitted = False
    retrieve_used = False

    def _build_lookup() -> dict[int, CitationRegistryEntry]:
        return {e.index: e for e in registry.values()}

    async def _execute_with_registry(name: str, args: dict) -> dict:
        return await execute_tool_fn(name, args, registry=registry, source_map=source_map)

    while True:
        iteration += 1
        tool_calls: list[ToolCall] = []
        iteration_text = ""
        lookup = _build_lookup()
        active_tools = [t for t in tools if not (retrieve_used and t.name == RETRIEVE_TOOL.name)]
        async for event in stream_llm(messages, cfg, active_tools, system=system, llm_max_tokens=llm_max_tokens):
            if event.type == "tool_call":
                tool_calls.append(event.tool_call)
            elif event.type == "delta" and event.content:
                iteration_text += event.content
            elif event.type == "done" and tool_calls:
                pass
            elif event.type in ("delta", "done"):
                pass
            else:
                yield event

        if not tool_calls:
            if iteration_text:
                parsed_events, parse_buffer = parse_delta(iteration_text, parse_buffer, lookup)
                for pe in parsed_events:
                    if pe.type == "citation":
                        citation_emitted = True
                    yield pe
            for pe in flush_buffer(parse_buffer):
                yield pe
            if not citation_emitted and registry:
                logger.info(
                    "citations_missing_after_retrieve registry_size=%d",
                    len(registry),
                )
            return

        parse_buffer = ""

        is_terminal = not any(tc.name in LOOPBACK_TOOLS for tc in tool_calls)
        if not is_terminal and iteration_text:
            logger.info("preamble_discarded len=%d iteration=%d", len(iteration_text), iteration)
        if is_terminal and iteration_text:
            parsed_events, _ = parse_delta(iteration_text, "", lookup)
            for pe in parsed_events:
                yield pe

        if iteration > MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="error", content="Max tool iterations exceeded")
            return

        results: dict[str, dict] = {}
        for tc in tool_calls:
            if tc.name in LOOPBACK_TOOLS:
                yield StreamEvent(
                    type="tool_call_start",
                    content=json.dumps({"id": tc.id, "name": tc.name, "arguments": tc.arguments}),
                )
        for tc in tool_calls:
            try:
                result = await _execute_with_registry(tc.name, tc.arguments)
            except Exception:
                logger.exception("tool_execution_failed tool=%s", tc.name)
                yield StreamEvent(type="error", content=f"Tool {tc.name} failed")
                return

            results[tc.id] = result
            yield StreamEvent(
                type="tool_result",
                content=json.dumps({"id": tc.id, "name": tc.name, "result": _client_tool_result(result)}),
            )

        if any(tc.name in LOOPBACK_TOOLS for tc in tool_calls):
            if any(tc.name == RETRIEVE_TOOL.name for tc in tool_calls):
                retrieve_used = True
            if cfg.protocol == "anthropic":
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                            for tc in tool_calls
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tc.id, "content": json.dumps(results[tc.id])}
                            for tc in tool_calls
                        ],
                    }
                )
            else:
                openai_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                messages.append({"role": "assistant", "content": None, "tool_calls": openai_tool_calls})
                for tc in tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(results[tc.id]),
                        }
                    )
            continue

        return


def _serialize_event_for_buffer(event: StreamEvent) -> dict | None:
    """Map a StreamEvent to the dict stored in StreamBuffer and sent via SSE."""
    if event.type == SSE_EVENT_DELTA:
        return {"type": SSE_EVENT_DELTA, "content": event.content or ""}
    elif event.type == SSE_EVENT_CITATION:
        data = json.loads(event.content)
        return {
            "type": SSE_EVENT_CITATION,
            "index": data["index"],
            "source_id": data["source_id"],
            "chunk_ids": data.get("chunk_ids", []),
        }
    elif event.type == SSE_EVENT_TOOL_CALL_START:
        parsed = json.loads(event.content)
        return {"type": SSE_EVENT_TOOL_CALL_START, **parsed}
    elif event.type == "tool_result":
        parsed = json.loads(event.content)
        return {"type": SSE_EVENT_TOOL_RESULT, **parsed}
    elif event.type == "error":
        return {"type": SSE_EVENT_ERROR, "message": event.content}
    return None


async def _sse_consumer(buf: StreamBuffer) -> AsyncGenerator[str, None]:
    async for event in stream_from_buffer(buf):
        yield f"data: {json.dumps(event)}\n\n"


async def _evict_after_grace(registry: ChatRunRegistry, message_id: str) -> None:
    await asyncio.sleep(STREAM_BUFFER_GRACE_SECONDS)
    registry.evict(message_id)


async def run_chat_turn(
    *,
    message_id: str,
    conversation_id: str,
    list_id: str,
    user_message_text: str,
    history: list[dict],
    summary: str | None,
    source_ids: list[str],
    source_map: dict[str, str],
    ui_lang: str,
    cfg: BibilabConfig,
    registry: ChatRunRegistry,
) -> None:
    buf = registry.get(message_id)
    if buf is None:
        logger.error("Buffer unexpectedly missing for message_id=%s", message_id)
        return

    final_status: TerminalStatus = "done"

    citation_registry: dict[str, CitationRegistryEntry] = {}
    assistant_text_deltas: list[str] = []
    retrieve_calls: list[dict] = []
    generate_report_result: dict | None = None
    content_blocks: list[dict] = []
    pending_text = ""

    try:
        system_parts = [GROUNDING_SYSTEM_PROMPT]
        if summary:
            system_parts.append(
                "Historical conversation summary (for context only — the current "
                "question may be about different sources than those summarized below):\n" + summary
            )
        system_message = "\n\n".join(system_parts)

        messages_for_llm = history + [{"role": "user", "content": user_message_text}]

        async def execute_tool_bound(name: str, args: dict, **kwargs) -> dict:
            return await execute_tool(
                tool_name=name,
                arguments=args,
                list_id=list_id,
                source_ids=source_ids,
                ui_lang=ui_lang,
                cfg=cfg,
                **kwargs,
            )

        tools = [RETRIEVE_TOOL, QUERY_LIST_METADATA_TOOL, GENERATE_REPORT_TOOL]

        async for event in stream_with_tools(
            messages=messages_for_llm,
            cfg=cfg.ai,
            tools=tools,
            execute_tool_fn=execute_tool_bound,
            system=system_message if system_message.strip() else None,
            llm_max_tokens=CHAT_MAX_TOKENS,
            registry=citation_registry,
            source_map=source_map,
        ):
            payload = _serialize_event_for_buffer(event)
            if payload is not None:
                buf.append(payload)

            if event.type == SSE_EVENT_DELTA:
                content = event.content or ""
                assistant_text_deltas.append(content)
                pending_text += content
            elif event.type == SSE_EVENT_CITATION:
                if pending_text:
                    _flush_pending_text(content_blocks, pending_text)
                    pending_text = ""
                data = json.loads(event.content)
                content_blocks.append(
                    {
                        "type": "citation",
                        "index": data["index"],
                        "source_id": data["source_id"],
                        "chunk_ids": data.get("chunk_ids", []),
                    }
                )
            elif event.type == "tool_result":
                parsed = json.loads(event.content)
                if parsed["name"] == "retrieve":
                    result = parsed["result"]
                    ordered_coverage = sorted(
                        [s for s in result.get("source_coverage", []) if s["source_id"] in citation_registry],
                        key=lambda s: citation_registry[s["source_id"]].index,
                    )
                    retrieve_calls.append(
                        {
                            "query": result.get("query", ""),
                            "search_mode": result.get("search_mode"),
                            "candidates_evaluated": result.get("candidates_evaluated"),
                            "sources_with_hits": result.get("sources_with_hits"),
                            "sources_total": result.get("sources_total"),
                            "source_coverage": ordered_coverage,
                        }
                    )
                elif parsed["name"] == "generate_report":
                    generate_report_result = parsed["result"]
            elif event.type == "error":
                logger.error("stream_with_tools error: %s", event.content)
                final_status = "failed"
                return
    except asyncio.CancelledError:
        final_status = "cancelled"
        raise
    except Exception:
        logger.exception("producer failed message_id=%s", message_id)
        final_status = "failed"
    finally:
        try:
            if pending_text:
                _flush_pending_text(content_blocks, pending_text)
                pending_text = ""

            tool_call_meta: list[dict] = []
            if generate_report_result is not None:
                tool_call_meta = [{"name": "generate_report", "result": generate_report_result}]

            meta: dict[str, Any] = {}
            if tool_call_meta:
                meta["tool_calls"] = tool_call_meta
            if retrieve_calls:
                meta["rag"] = {"calls": retrieve_calls}
            if content_blocks:
                meta["content_blocks"] = content_blocks

            assistant_content = "".join(assistant_text_deltas)
            error_text = "An internal error occurred" if final_status == "failed" else None
            await update_message_content(
                message_id,
                content=assistant_content,
                metadata=meta if meta else None,
                status=final_status,
                error=error_text,
            )
        except Exception:
            logger.exception("producer finalize failed message_id=%s", message_id)

        try:
            await set_active_stream(conversation_id, None)
        except Exception:
            logger.exception("producer clear active_stream failed message_id=%s", message_id)

        _terminal_map = {"done": SSE_EVENT_DONE, "cancelled": SSE_EVENT_CANCELLED, "failed": SSE_EVENT_ERROR}
        sse_terminal = _terminal_map[final_status]
        terminal_payload: dict[str, Any] = {"type": sse_terminal}
        if final_status == "failed":
            terminal_payload["message"] = "An internal error occurred"
        buf.append(terminal_payload)
        buf.close(final_status)

        registry.track_background(asyncio.create_task(_evict_after_grace(registry, message_id)))

        if final_status == "done":
            registry.track_background(asyncio.create_task(maybe_compress_conversation(conversation_id, cfg)))


@router.post("/lists/{list_id}/chat")
async def chat_endpoint(
    list_id: str,
    request: ChatRequest,
    http_request: Request,
    cfg: BibilabConfig = Depends(get_config),
    run_registry: ChatRunRegistry = Depends(get_chat_run_registry),
) -> StreamingResponse:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    conversation_id = await get_or_create_conversation(list_id)

    conv_row = await get_conv_row(conversation_id)
    existing_summary = conv_row["summary"] if conv_row else None

    # Snapshot history before inserting new messages — the producer adds the
    # current user message explicitly via user_message_text.
    history_rows = await get_recent_messages(conversation_id, limit=100)
    history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    source_rows = await get_sources_for_list(list_id)
    if request.source_ids:
        source_ids = request.source_ids
    else:
        source_ids = [row["id"] for row in source_rows]

    # Atomic 409 guard + insert user msg + insert streaming assistant msg
    user_msg_id = str(uuid4())
    assistant_msg_id = str(uuid4())
    try:
        await create_user_and_assistant_atomic(
            conversation_id=conversation_id,
            user_msg_id=user_msg_id,
            assistant_msg_id=assistant_msg_id,
            user_text=request.message,
        )
    except ActiveStreamConflict:
        raise HTTPException(409, "Conversation already has an active stream")

    source_map: dict[str, str] = {row["video_id"]: row["id"] for row in source_rows}
    ui_lang = http_request.headers.get("X-UI-Lang", "en")

    # Spawn producer
    task = asyncio.create_task(
        run_chat_turn(
            message_id=assistant_msg_id,
            conversation_id=conversation_id,
            list_id=list_id,
            user_message_text=request.message,
            history=history,
            summary=existing_summary,
            source_ids=source_ids,
            source_map=source_map,
            ui_lang=ui_lang,
            cfg=cfg,
            registry=run_registry,
        )
    )
    buf = run_registry.register(assistant_msg_id, task)
    # Let client know the server-assigned id so Stop can target it before the
    # first delta arrives.  Appended to the buffer before the consumer starts
    # reading, so it is always the first event delivered.
    buf.append({"type": "meta", "message_id": assistant_msg_id})

    return StreamingResponse(
        _sse_consumer(buf),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.get("/lists/{list_id}/chat/{message_id}/stream")
async def reattach_stream(
    list_id: str,
    message_id: str,
    run_registry: ChatRunRegistry = Depends(get_chat_run_registry),
):
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(404, "List not found")

    if not await assert_message_in_list(message_id, list_id):
        raise HTTPException(404, "Message not in list")

    buf = run_registry.get(message_id)
    if buf is None:
        return Response(status_code=204)

    return StreamingResponse(
        _sse_consumer(buf),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/lists/{list_id}/chat/{message_id}/cancel", status_code=204)
async def cancel_stream(
    list_id: str,
    message_id: str,
    run_registry: ChatRunRegistry = Depends(get_chat_run_registry),
):
    if not await assert_message_in_list(message_id, list_id):
        raise HTTPException(404, "Message not in list")
    run_registry.cancel(message_id)
