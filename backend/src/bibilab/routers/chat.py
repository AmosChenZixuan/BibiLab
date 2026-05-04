import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from bibilab.config import AIConfig, BibilabConfig, get_config
from bibilab.db import (
    create_message,
    delete_conversation,
    delete_messages_by_ids,
    get_conversation_by_list,
    get_list,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_for_list,
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
from bibilab.pipeline.chat_summary import maybe_compress_conversation
from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL, RETRIEVE_TOOL, execute_tool

logger = logging.getLogger(__name__)

router = APIRouter()

# SSE event types — used both as stream-internal event discriminators (stream_llm yield)
# and as the 'type' field in the SSE 'data:' JSON payload sent to the client.
SSE_EVENT_DELTA = "delta"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"
SSE_EVENT_TOOL_RESULT = "tool_result"

# Sized for thinking-capable models with potentially long chat responses + tool turns.
CHAT_MAX_TOKENS = 16384

LOOPBACK_TOOLS = {"retrieve"}
MAX_TOOL_ITERATIONS = 3


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


def _client_tool_result(name: str, result: dict) -> dict:
    """Strip internal fields before sending to client."""
    if name == "retrieve":
        return {k: v for k, v in result.items() if k != "_chunks"}
    return result


GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly based on the provided source material. "
    "CRITICAL RULES:\n"
    "0. If the user's question requires looking up video content (facts, comparisons, "
    "summaries, counts across sources), you MUST call the retrieve tool BEFORE answering. "
    "Do not answer from memory — always retrieve first for content questions. "
    "This does NOT apply when the user asks you to generate a report/artifact — "
    "the generate_report tool handles its own retrieval.\n"
    "1. ONLY use information from the source material provided below. Never use your own knowledge.\n"
    '2. If the excerpts do not contain the answer, say "The provided sources do not cover this topic."\n'
    "3. Quote or closely paraphrase the source material — do not reinterpret, editorialize, or add external context.\n"
    "4. Cite sources using EXACTLY this format: [video_title @ Ns-Ns] — e.g. [My Video @ 120s-145s].\n"
    "5. Use the generate_report tool when the user asks for summaries, study guides, blog posts, or custom reports.\n"
    "6. Do not ask follow-up questions, suggest next steps, or offer unsolicited advice.\n"
    "7. Be concise and direct. Answer in 1-3 sentences when possible."
)


async def stream_with_tools(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition],
    execute_tool_fn,
    system: str | None = None,
    llm_max_tokens: int = CHAT_MAX_TOKENS,
) -> AsyncGenerator[StreamEvent, None]:
    """Call stream_llm in a loop. Loopback tools (retrieve) feed results back
    for another LLM turn. Terminal tools (generate_report) exit the loop."""
    messages = list(messages)
    iteration = 0

    while True:
        iteration += 1
        tool_calls: list[ToolCall] = []
        async for event in stream_llm(messages, cfg, tools, system=system, llm_max_tokens=llm_max_tokens):
            if event.type == "tool_call":
                tool_calls.append(event.tool_call)
            yield event

        if not tool_calls:
            return

        if iteration > MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="error", content="Max tool iterations exceeded")
            return

        results: dict[str, dict] = {}
        for tc in tool_calls:
            try:
                result = await execute_tool_fn(tc.name, tc.arguments)
                results[tc.id] = result
            except Exception as exc:
                yield StreamEvent(type="error", content=str(exc))
                return

            yield StreamEvent(
                type="tool_result",
                content=json.dumps({"name": tc.name, "result": _client_tool_result(tc.name, result)}),
            )

        # LOOPBACK_TOOLS = {"retrieve"}. If the LLM calls both retrieve and a
        # terminal tool (generate_report) in the same turn, the loopback branch
        # still runs — the terminal tool's messages are fed back to the LLM too.
        # In practice the system prompt steers them apart; if this ever causes
        # issues, check for terminal-tool membership here and exit early.
        if any(tc.name in LOOPBACK_TOOLS for tc in tool_calls):
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


@router.post("/lists/{list_id}/chat")
async def chat_endpoint(
    list_id: str,
    request: ChatRequest,
    http_request: Request,
    cfg: BibilabConfig = Depends(get_config),
) -> StreamingResponse:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    conversation_id = await get_or_create_conversation(list_id)

    conv_row = await get_conv_row(conversation_id)
    existing_summary = conv_row["summary"] if conv_row else None

    history_rows = await get_recent_messages(conversation_id, limit=100)
    history = [{"role": row["role"], "content": row["content"]} for row in history_rows]

    if request.source_ids:
        source_ids = request.source_ids
    else:
        source_rows = await get_sources_for_list(list_id)
        source_ids = [row["id"] for row in source_rows]

    ui_lang = http_request.headers.get("X-UI-Lang", "en")

    user_msg_row = await create_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        metadata=None,
    )
    user_msg_id = user_msg_row["id"]

    first_response_deltas: list[str] = []
    tool_calls: list = []
    retrieve_result: dict | None = None
    generate_report_result: dict | None = None

    async def event_generator():
        nonlocal first_response_deltas, tool_calls, retrieve_result, generate_report_result

        system_parts = [GROUNDING_SYSTEM_PROMPT]
        if existing_summary:
            system_parts.append(
                "Historical conversation summary (for context only — the current "
                "question may be about different sources than those summarized below):\n" + existing_summary
            )
        system_message = "\n\n".join(system_parts)

        messages_for_llm = history + [{"role": "user", "content": request.message}]

        async def execute_tool_bound(name: str, args: dict) -> dict:
            return await execute_tool(
                tool_name=name,
                arguments=args,
                list_id=list_id,
                source_ids=source_ids,
                ui_lang=ui_lang,
                cfg=cfg,
            )

        tools = [RETRIEVE_TOOL, GENERATE_REPORT_TOOL]

        try:
            async for event in stream_with_tools(
                messages=messages_for_llm,
                cfg=cfg.ai,
                tools=tools,
                execute_tool_fn=execute_tool_bound,
                system=system_message if system_message.strip() else None,
                llm_max_tokens=CHAT_MAX_TOKENS,
            ):
                if event.type == SSE_EVENT_DELTA:
                    first_response_deltas.append(event.content or "")
                    yield f"data: {json.dumps({'type': SSE_EVENT_DELTA, 'content': event.content})}\n\n"
                elif event.type == "tool_call":
                    # Collect tool_calls locally to gate the done-event suppression
                    # and metadata persistence below. stream_with_tools has its own
                    # per-iteration list that drives the loopback decision — the two
                    # lists are intentionally independent.
                    tool_calls.append(event.tool_call)
                elif event.type == "tool_result":
                    parsed = json.loads(event.content)
                    if parsed["name"] == "retrieve":
                        # Last retrieve result wins for metadata persistence.
                        # In practice the LLM rarely calls retrieve twice per
                        # turn, and the second call is usually the more refined
                        # one (narrower query after seeing first results).
                        retrieve_result = parsed["result"]
                    elif parsed["name"] == "generate_report":
                        generate_report_result = parsed["result"]
                    yield f"data: {json.dumps({'type': SSE_EVENT_TOOL_RESULT, **parsed})}\n\n"
                elif event.type == SSE_EVENT_DONE:
                    if not tool_calls:
                        yield f"data: {json.dumps({'type': SSE_EVENT_DONE})}\n\n"
                elif event.type == "error":
                    logger.error("stream_with_tools error: %s", event.content)
                    await delete_messages_by_ids([user_msg_id])
                    yield f"data: {json.dumps({'type': SSE_EVENT_ERROR, 'message': event.content})}\n\n"
                    return
        except Exception:
            logger.exception("LLM streaming failed")
            await delete_messages_by_ids([user_msg_id])
            yield f"data: {json.dumps({'type': SSE_EVENT_ERROR, 'message': 'An internal error occurred'})}\n\n"
            return

        if not tool_calls:
            full_response = "".join(first_response_deltas)
            await create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                metadata=None,
            )
            return

        # Persist tool call metadata (results already collected during stream).
        # generate_report includes its result so the frontend can rehydrate
        # artifact links from history without a live SSE tool_result event.
        tool_call_meta: list[dict] = []
        for tc in tool_calls:
            entry: dict[str, Any] = {"id": tc.id, "name": tc.name}
            if tc.name == "generate_report" and generate_report_result:
                entry["result"] = generate_report_result
            tool_call_meta.append(entry)

        meta: dict[str, Any] = {}
        if tool_call_meta:
            meta["tool_calls"] = tool_call_meta
        if retrieve_result:
            meta["rag"] = {
                "search_mode": retrieve_result.get("search_mode"),
                "candidates_evaluated": retrieve_result.get("candidates_evaluated"),
                "sources_with_hits": retrieve_result.get("sources_with_hits"),
                "sources_total": retrieve_result.get("sources_total"),
                "source_coverage": retrieve_result.get("source_coverage"),
            }

        # Persist any preamble + post-loopback deltas alongside tool metadata
        # so the full response survives history reload (not just live SSE).
        assistant_content = "".join(first_response_deltas) if first_response_deltas else ""
        await create_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
            metadata=meta if meta else None,
        )

        yield f"data: {json.dumps({'type': SSE_EVENT_DONE})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
        background=BackgroundTask(maybe_compress_conversation, conversation_id, cfg),
    )
