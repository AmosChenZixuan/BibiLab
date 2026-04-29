import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from bibilab.config import BibilabConfig, get_config
from bibilab.db import (
    create_message,
    delete_conversation,
    delete_messages_by_ids,
    get_conversation_by_list,
    get_list,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_for_list,
    log_query_classification,
    update_conversation_mode,
)
from bibilab.db import (
    get_conversation as get_conv_row,
)
from bibilab.models._enums import CHAT_MODE_BROAD, CHAT_MODE_FOCUSED
from bibilab.models.chat import (
    ChatRequest,
    ConversationResponse,
    GetConversationResponse,
    MessageResponse,
    PatchConversationRequest,
)
from bibilab.pipeline._shared import stream_llm
from bibilab.pipeline.chat_summary import maybe_compress_conversation
from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL, execute_tool
from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, retrieve
from bibilab.pipeline.route import classify_query, map_type_to_mode

logger = logging.getLogger(__name__)

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


@router.patch("/lists/{list_id}/conversation", response_model=ConversationResponse)
async def patch_conversation(list_id: str, request: PatchConversationRequest) -> ConversationResponse:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")
    conversation_id = await get_or_create_conversation(list_id)
    await update_conversation_mode(conversation_id, request.mode)
    conv_row = await get_conv_row(conversation_id)
    return ConversationResponse.from_row(dict(conv_row))


def _format_chunk_line(chunk: RetrievedChunk) -> str:
    ts_start = int(chunk.timestamp_start)
    ts_end = int(chunk.timestamp_end)
    return f'- [{chunk.video_title} @ {ts_start}s-{ts_end}s]: "{chunk.content}"'


def _format_rag_context(result: RetrievalResult, query: str) -> str:
    if not result.chunks:
        return ""
    if result.mode == CHAT_MODE_BROAD:
        parts = [
            f"Query: {query}\n",
            f"Concept appears in {result.sources_with_hits} of {result.sources_total} sources\n",
            "Best excerpt per source:",
        ]
        parts.extend(_format_chunk_line(c) for c in result.chunks)
        return "\n".join(parts)
    parts = [f"Query: {query}\n\nRelevant transcript excerpts:"]
    parts.extend(_format_chunk_line(c) for c in result.chunks)
    return "\n".join(parts)


def _build_rag_payload(rag_result: RetrievalResult) -> dict:
    return {
        "type": "rag_meta",
        "rag": {
            "mode": rag_result.mode,
            "candidates_evaluated": rag_result.candidates_evaluated,
            "sources_with_hits": rag_result.sources_with_hits,
            "sources_total": rag_result.sources_total,
            "sources": [{"video_id": sh.video_id, "title": sh.video_title} for sh in rag_result.source_coverage],
        },
    }


GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly based on the provided transcript excerpts. "
    "CRITICAL RULES:\n"
    "1. ONLY use information from the transcript excerpts provided below. Never use your own knowledge.\n"
    '2. If the excerpts do not contain the answer, say "The provided transcripts do not cover this topic."\n'
    "3. Quote or closely paraphrase the transcript — do not reinterpret, editorialize, or add external context.\n"
    "4. Cite sources using EXACTLY this format: [video_title @ Ns-Ns] — e.g. [My Video @ 120s-145s].\n"
    "5. Use the generate_report tool when the user asks for summaries, study guides, blog posts, or custom reports.\n"
    "6. Do not ask follow-up questions, suggest next steps, or offer unsolicited advice.\n"
    "7. Be concise and direct. Answer in 1-3 sentences when possible."
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

    conv_row = await get_conv_row(conversation_id)
    existing_summary = conv_row["summary"] if conv_row else None
    conversation_mode = conv_row["mode"] if conv_row else CHAT_MODE_FOCUSED

    history_rows = await get_recent_messages(conversation_id, limit=100)
    history = [{"role": row["role"], "content": row["content"]} for row in history_rows]

    if request.source_ids:
        source_ids = request.source_ids
    else:
        source_rows = await get_sources_for_list(list_id)
        source_ids = [row["id"] for row in source_rows]

    effective_mode = conversation_mode
    rag_context = ""
    rag_result = None
    if source_ids and request.message.strip():
        if cfg.rag.query_routing_enabled and conversation_mode == CHAT_MODE_FOCUSED:
            query_type = await classify_query(request.message, cfg)
            effective_mode = map_type_to_mode(query_type)
            await log_query_classification(
                list_id=list_id,
                query_text=request.message,
                query_type=query_type,
                effective_mode=effective_mode,
            )
        rag_result = await retrieve(
            query_text=request.message,
            source_ids=source_ids,
            cfg=cfg,
            top_k=5,
            mode=effective_mode,
        )
        rag_context = _format_rag_context(rag_result, request.message)

    system_parts = [GROUNDING_SYSTEM_PROMPT]
    if existing_summary:
        system_parts.append(f"\n\nEarlier conversation summary:\n{existing_summary}")
    if rag_context:
        system_parts.append(rag_context)
    system_message = "\n\n".join(system_parts)

    messages_for_llm = history + [{"role": "user", "content": request.message}]

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

    async def event_generator():
        nonlocal first_response_deltas, tool_calls

        rag_payload_obj = _build_rag_payload(rag_result) if rag_result and rag_result.chunks else None
        rag_meta_inner = rag_payload_obj["rag"] if rag_payload_obj else None

        if rag_payload_obj is not None:
            yield f"data: {json.dumps(rag_payload_obj)}\n\n"

        try:
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
                    if not tool_calls:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            logger.exception("LLM streaming failed")
            await delete_messages_by_ids([user_msg_id])
            yield f"data: {json.dumps({'type': 'error', 'message': 'An internal error occurred'})}\n\n"
            return

        if not tool_calls:
            full_response = "".join(first_response_deltas)
            metadata = {"rag": rag_meta_inner} if rag_meta_inner else None
            await create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                metadata=metadata,
            )
            return

        yield f"data: {json.dumps({'type': 'clear_text'})}\n\n"

        tool_results = []
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

            tool_results.append({"tool_call_id": tc.id, "content": json.dumps({"name": tc.name, "result": result})})
            yield f"data: {json.dumps({'type': 'tool_result', 'tool_call_id': tc.id, 'result': result})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        tool_call_meta = []
        for tc, tr in zip(tool_calls, tool_results):
            result_data = json.loads(tr["content"])
            tool_call_meta.append({"id": tc.id, "name": tc.name, "result": result_data.get("result")})

        meta: dict[str, Any] = {"tool_calls": tool_call_meta}
        if rag_meta_inner is not None:
            meta["rag"] = rag_meta_inner
        await create_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            metadata=meta,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
        background=BackgroundTask(maybe_compress_conversation, conversation_id, cfg),
    )
