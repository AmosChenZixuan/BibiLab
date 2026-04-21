import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from bibilab.config import BibilabConfig, get_config
from bibilab.db import (
    create_message,
    delete_conversation,
    get_conversation_by_list,
    get_list,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_for_list,
)
from bibilab.models.chat import (
    ChatRequest,
    ConversationResponse,
    GetConversationResponse,
    MessageResponse,
)
from bibilab.pipeline._shared import stream_llm
from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL, execute_tool
from bibilab.pipeline.embed import query_chunks

router = APIRouter()


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


def _format_rag_context(chunks: list, query: str) -> str:
    if not chunks:
        return ""
    parts = [f"Query: {query}\n\nRelevant transcript excerpts:"]
    for chunk in chunks:
        ts_start = int(chunk.timestamp_start)
        ts_end = int(chunk.timestamp_end)
        parts.append(f'- [{chunk.video_title}] @ {ts_start}s-{ts_end}s: "{chunk.content}"')
    return "\n".join(parts)


GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about video content. "
    "Answer based on the provided transcript excerpts when available. "
    "Cite specific timestamps when referencing content from transcripts using the format [video_title @ Ts-Ts]. "
    "Use the generate_report tool when the user asks for summaries, study guides, blog posts, or custom reports."
)


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

    history_rows = await get_recent_messages(conversation_id, limit=100)
    history = [{"role": row["role"], "content": row["content"]} for row in history_rows]

    source_rows = await get_sources_for_list(list_id)
    source_ids = [row["id"] for row in source_rows]

    rag_chunks = []
    rag_context = ""
    if source_ids and request.message.strip():
        rag_chunks = await query_chunks(
            query_text=request.message,
            source_ids=source_ids,
            cfg=cfg,
            top_k=5,
        )
        rag_context = _format_rag_context(rag_chunks, request.message)

    system_parts = [GROUNDING_SYSTEM_PROMPT]
    if rag_context:
        system_parts.append("\n\n" + rag_context)
    system_message = "\n".join(system_parts)

    messages_for_llm = history + [{"role": "user", "content": request.message}]

    ui_lang = http_request.headers.get("X-UI-Lang", "en")

    await create_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        metadata=None,
    )

    first_response_deltas: list[str] = []
    tool_calls: list = []

    async def event_generator():
        nonlocal first_response_deltas, tool_calls

        async for event in stream_llm(
            messages=messages_for_llm,
            cfg=cfg.ai,
            tools=[GENERATE_REPORT_TOOL],
            system=system_message if system_message.strip() else None,
        ):
            if event.type == "delta":
                first_response_deltas.append(event.content or "")
                yield f"data: {json.dumps({'type': 'delta', 'content': event.content})}\n\n"
            elif event.type == "tool_call":
                tool_calls.append(event.tool_call)
            elif event.type == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        if not tool_calls:
            full_response = "".join(first_response_deltas)
            await create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                metadata=None,
            )
            return

        tool_messages = []
        for tc in tool_calls:
            try:
                result = await execute_tool(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    list_id=list_id,
                    source_ids=source_ids,
                    ui_lang=ui_lang,
                )
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                full_response = "".join(first_response_deltas)
                await create_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                    metadata={"tool_calls": [{"id": tc.id, "name": tc.name, "error": str(exc)}]},
                )
                return

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"name": tc.name, "result": result}),
                }
            )
            yield f"data: {json.dumps({'type': 'tool_result', 'tool_call_id': tc.id, 'result': result})}\n\n"

        second_messages = messages_for_llm + [
            {"role": "assistant", "content": "".join(first_response_deltas)},
            *tool_messages,
        ]

        second_deltas: list[str] = []

        async for event in stream_llm(
            messages=second_messages,
            cfg=cfg.ai,
            system=system_message if system_message.strip() else None,
        ):
            if event.type == "delta":
                second_deltas.append(event.content or "")
                yield f"data: {json.dumps({'type': 'delta', 'content': event.content})}\n\n"
            elif event.type == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        full_response = "".join(second_deltas)
        await create_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            metadata={"tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]},
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
