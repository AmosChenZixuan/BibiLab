import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from bibilab.config import BibilabConfig, bibilab_home, get_config
from bibilab.db import (
    VISIBLE_MESSAGE_STATUS,
    assert_message_in_list,
    create_artifact,
    delete_conversation,
    get_artifact,
    get_conversation_by_list,
    get_list,
    get_message,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_by_ids,
    get_sources_for_list,
    get_user_prompt_for_assistant,
    set_active_stream,
)
from bibilab.db import (
    get_conversation as get_conv_row,
)
from bibilab.models.artifacts import ArtifactResponse
from bibilab.models.chat import (
    ChatRequest,
    ChatSaveMessageRequest,
    ConversationResponse,
    GetConversationResponse,
    MessageResponse,
)
from bibilab.pipeline._shared import (
    UI_LANG_HEADER,
    StreamEvent,
    ToolDefinition,
    _classify_llm_error,
    format_mmss,
    resolve_response_language,
)
from bibilab.pipeline.chat_ledger import build_rag_ledger
from bibilab.pipeline.chat_loop import (
    ERROR_CODE_PERSISTENCE,
    ERROR_CODE_TOOL,
    SSE_EVENT_CANCELLED,
    SSE_EVENT_CITATION,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_EVENT_META,
    SSE_EVENT_RAG,
    SSE_EVENT_TOOL_CALL_START,
    SSE_EVENT_TOOL_RESULT,
    build_grounding_prompt,
    stream_with_tools,
)
from bibilab.pipeline.chat_runs import (
    STREAM_BUFFER_GRACE_SECONDS,
    ActiveStreamConflict,
    ChatRunRegistry,
    StreamBuffer,
    TerminalStatus,
    create_user_and_assistant_atomic,
    get_chat_run_registry,
    stream_from_buffer,
    update_turn_terminal,
)
from bibilab.pipeline.chat_summary import maybe_compress_conversation
from bibilab.pipeline.chat_tools import (
    FIND_PASSAGES_TOOL,
    READ_SECTION_TOOL,
    RETRIEVE_TOOL_NAMES,
    CitationRegistryEntry,
    execute_tool,
    expand_message_for_provider,
    reseed_citation_registry,
)
from bibilab.routers._model_gate import require_models_present

logger = logging.getLogger(__name__)

router = APIRouter()
debug_router = APIRouter()


@router.get("/lists/{list_id}/conversation")
async def get_conversation(
    list_id: str,
    before: str | None = None,
    limit: int = 50,
    cfg: BibilabConfig = Depends(get_config),
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

    messages = [MessageResponse.from_row(dict(r)) for r in messages_rows]

    # Only scan the debug dir when prompt-trace dumps are enabled. Off (the
    # default) means no dumps are ever written, so has_dump must be False.
    if cfg.rag.debug_prompts:
        debug_dir = bibilab_home() / "debug"
        existing = {p.stem for p in debug_dir.glob("*.json")} if debug_dir.exists() else set()
        for m in messages:
            m.has_dump = m.id in existing

    return GetConversationResponse(
        conversation=ConversationResponse.from_row(dict(conversation_row)),
        messages=messages,
    )


@router.delete("/lists/{list_id}/conversation", status_code=204)
async def delete_conversation_endpoint(list_id: str) -> None:
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    conversation_row = await get_conversation_by_list(list_id)
    if conversation_row is not None:
        await delete_conversation(conversation_row["id"])


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


def _serialize_event_for_buffer(event: StreamEvent) -> dict | None:
    """Map a StreamEvent to the dict stored in StreamBuffer and sent via SSE."""
    if event.type == SSE_EVENT_DELTA:
        return {"type": SSE_EVENT_DELTA, "content": event.content or ""}
    elif event.type == SSE_EVENT_CITATION:
        data = json.loads(event.content)
        return {
            "type": SSE_EVENT_CITATION,
            "index": data["index"],
            "section_id": data.get("section_id", ""),
            "source_id": data["source_id"],
            "timestamp_start": data.get("timestamp_start", 0.0),
            "chunk_ids": data.get("chunk_ids", []),
        }
    elif event.type == SSE_EVENT_TOOL_CALL_START:
        parsed = json.loads(event.content)
        return {"type": SSE_EVENT_TOOL_CALL_START, **parsed}
    elif event.type == SSE_EVENT_TOOL_RESULT:
        parsed = json.loads(event.content)
        return {"type": SSE_EVENT_TOOL_RESULT, **parsed}
    elif event.type == SSE_EVENT_ERROR:
        return {"type": SSE_EVENT_ERROR, "message": event.content}
    return None


def _dump_turn(
    debug_path: Path,
    *,
    system: str | None,
    messages: list[dict],
    tools: list[ToolDefinition],
    response_text: str = "",
    model: str = "",
    timestamp: str = "",
) -> None:
    """Best-effort write of one chat turn's final LLM state.

    `debug_path` is the full file path (e.g. `~/.bibilab/debug/{message_id}.json`),
    not a directory. Writes {system, tools, messages, response: {text},
    model, timestamp} verbatim as JSON. The final LLM call's `messages` is the
    cumulative state — it already contains all prior tool results — so one file
    per message captures the final state the LLM actually saw. All errors are
    caught and logged; a dump failure must never break a turn.
    """
    try:
        payload = {
            "system": system,
            "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools],
            "messages": messages,
            "response": {
                "text": response_text,
            },
            "model": model,
            "timestamp": timestamp,
        }
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        logger.warning("dump_turn_failed path=%s", debug_path, exc_info=True)


async def _sse_consumer(buf: StreamBuffer) -> AsyncGenerator[str, None]:
    try:
        async for event in stream_from_buffer(buf):
            yield f"data: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        # Client disconnected — normal, no action needed.
        raise
    except Exception:
        logger.exception("SSE consumer failed message_id=%s", buf.message_id)


async def _evict_after_grace(registry: ChatRunRegistry, message_id: str) -> None:
    await asyncio.sleep(STREAM_BUFFER_GRACE_SECONDS)
    registry.evict(message_id)


async def run_chat_turn(
    *,
    message_id: str,
    conversation_id: str,
    user_message_text: str,
    history: list[dict],
    summary: str | None,
    source_ids: list[str],
    ui_lang: str,
    cfg: BibilabConfig,
    registry: ChatRunRegistry,
    user_msg_id: str,
) -> None:
    buf = registry.get(message_id)
    if buf is None:
        logger.error("Buffer unexpectedly missing for message_id=%s", message_id)
        return

    final_status: TerminalStatus = "done"

    citation_registry: dict[str, CitationRegistryEntry] = {}
    assistant_text_deltas: list[str] = []
    retrieve_calls: list[dict] = []
    read_section_calls: list[dict] = []
    all_calls: list[dict] = []
    content_blocks: list[dict] = []
    pending_text = ""
    error_reason: str | None = None

    try:
        response_language = resolve_response_language(cfg.ai, ui_lang)
        system_parts = [build_grounding_prompt(response_language=response_language)]
        if summary:
            system_parts.append(
                "Historical conversation summary (for context only — the current "
                "question may be about different sources than those summarized below):\n" + summary
            )
        system_message = "\n\n".join(system_parts)

        # Prior retrieve excerpts replay into the LLM messages; the LLM self-
        # judges reuse from the grounding prompt instruction. Pure acks are
        # handled by the LLM choosing not to call any retrieve tool — there is
        # no deterministic strip step.
        history_for_expansion = history

        # Reseed citation registry and expand history tool blocks for replay.
        reseed_citation_registry(citation_registry, history_for_expansion)
        expanded_history: list[dict] = []
        for h in history_for_expansion:
            expanded_history.extend(expand_message_for_provider(h, protocol=cfg.ai.protocol))
        messages_for_llm = expanded_history + [{"role": "user", "content": user_message_text}]

        async def execute_tool_bound(name: str, args: dict, **kwargs) -> dict:
            return await execute_tool(
                tool_name=name,
                arguments=args,
                source_ids=source_ids,
                cfg=cfg,
                **kwargs,
            )

        tools = [FIND_PASSAGES_TOOL, READ_SECTION_TOOL]

        tool_blocks: list[dict] = []
        # Cumulative LLM message list at end-of-turn. stream_with_tools rebinds
        # messages to a defensive local copy, so in-loop appends never reach
        # messages_for_llm — the sink captures the final state instead.
        final_messages: list[dict] = []

        async for event in stream_with_tools(
            messages=messages_for_llm,
            cfg=cfg.ai,
            tools=tools,
            execute_tool_fn=execute_tool_bound,
            system=system_message if system_message.strip() else None,
            registry=citation_registry,
            tool_block_sink=tool_blocks,
            messages_sink=final_messages,
            response_language=response_language,
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
                        "section_id": data.get("section_id", ""),
                        "source_id": data["source_id"],
                        "timestamp_start": data.get("timestamp_start", 0.0),
                        "chunk_ids": data.get("chunk_ids", []),
                    }
                )
            elif event.type == SSE_EVENT_TOOL_CALL_START:
                # Flush preamble + paragraph break; idempotent, mirrored in useSSEStream.ts.
                if pending_text:
                    _flush_pending_text(content_blocks, pending_text)
                    pending_text = ""
                if content_blocks and content_blocks[-1].get("type") != "paragraph_break":
                    content_blocks.append({"type": "paragraph_break"})
            elif event.type == SSE_EVENT_TOOL_RESULT:
                parsed = json.loads(event.content)
                if parsed["name"] in RETRIEVE_TOOL_NAMES:
                    result = parsed["result"]
                    # Store raw section_coverage for now; narrowed by emitted citations in build_rag_ledger.
                    retrieve_calls.append(
                        {
                            "query": result.get("query", ""),
                            "tool_name": result.get("tool_name", parsed["name"]),
                            "candidates_evaluated": result.get("candidates_evaluated"),
                            "sources_with_hits": result.get("sources_with_hits"),
                            "sources_total": result.get("sources_total"),
                            "section_coverage": result.get("section_coverage", []),
                            "reranked": result.get("reranked", False),
                            "scoped_pool_size": result.get("scoped_pool_size"),
                            "facet_scope": result.get("facet_scope"),
                        }
                    )
                elif parsed["name"] == READ_SECTION_TOOL.name:
                    sid = parsed["result"].get("source_id")
                    if sid:  # None on a resolution error → nothing was read, no ledger row
                        read_section_calls.append(
                            {
                                "tool_name": READ_SECTION_TOOL.name,
                                "section_id": parsed["result"].get("section_id", ""),
                                "source_id": sid,
                                "source_title": parsed["result"].get("source_title", ""),
                            }
                        )
            elif event.type == SSE_EVENT_ERROR:
                logger.error("stream_with_tools error: %s", event.content)
                error_reason = ERROR_CODE_TOOL
                final_status = "failed"
                return

        # End-of-turn dump: write one file capturing the final LLM state.
        # final_messages is the cumulative list exported by stream_with_tools
        # via messages_sink — it already includes all prior tool exchanges, so
        # a single file per turn replaces the older one-file-per-llm-call
        # scheme. We can't use messages_for_llm directly: stream_with_tools
        # rebinds messages to a defensive local copy, so in-loop appends
        # never propagate back here.
        if cfg.rag.debug_prompts:
            debug_path = bibilab_home() / "debug" / f"{message_id}.json"
            _dump_turn(
                debug_path,
                system=system_message if system_message.strip() else None,
                messages=final_messages,
                tools=tools,
                response_text="".join(assistant_text_deltas),
                model=cfg.ai.model,
                timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
    except asyncio.CancelledError:
        final_status = "cancelled"
        raise
    except Exception as e:
        logger.exception("producer failed message_id=%s", message_id)
        error_reason = _classify_llm_error(e)
        final_status = "failed"
    finally:
        try:
            if pending_text:
                _flush_pending_text(content_blocks, pending_text)
                pending_text = ""

            all_calls = build_rag_ledger(
                retrieve_calls=retrieve_calls,
                read_section_calls=read_section_calls,
                content_blocks=content_blocks,
                citation_registry=citation_registry,
            )

            meta: dict[str, Any] = {}

            if all_calls:
                meta["rag"] = {"calls": all_calls}
            if content_blocks:
                meta["content_blocks"] = content_blocks

            assistant_content = "".join(assistant_text_deltas)
            error_text = error_reason if error_reason else ("internal_error" if final_status == "failed" else None)
            # Atomically flip both rows of the turn to the same terminal
            # status AND clear active_stream_message_id. The user row only
            # changes status+error (content/metadata/tool_blocks are unchanged
            # from insert time); all three writes commit together so a process
            # kill cannot strand an orphan or leave a wedged 409 pointer.
            await update_turn_terminal(
                conversation_id=conversation_id,
                user_msg_id=user_msg_id,
                asst_msg_id=message_id,
                asst_content=assistant_content,
                asst_metadata=meta if meta else None,
                asst_tool_blocks=tool_blocks if tool_blocks else None,
                status=final_status,
                error=error_text,
            )
        except Exception:
            logger.exception("producer finalize failed message_id=%s", message_id)
            # Don't clobber a "cancelled" status set by the asyncio.CancelledError
            # branch — the SSE event must reflect the user's action, not a
            # downstream persistence hiccup.
            if final_status != "cancelled":
                final_status = "failed"
            if error_reason is None:
                error_reason = ERROR_CODE_PERSISTENCE
            # update_turn_terminal's transaction rolled back, so active_stream_message_id
            # was never cleared. Clear it independently or the conversation wedges at
            # HTTP 409 (the guard checks only pointer-not-null) until the next restart.
            try:
                await set_active_stream(conversation_id, None)
            except Exception:
                logger.exception(
                    "producer fallback clear active_stream failed message_id=%s — "
                    "stale pointer may 409 future POSTs until restart",
                    message_id,
                )

        # Final authoritative ledger: persisted-shape calls (context[]
        # reconstructed) so the client matches post-refresh without reloading.
        if all_calls:
            buf.append({"type": SSE_EVENT_RAG, "calls": all_calls})

        terminal_map = {"done": SSE_EVENT_DONE, "cancelled": SSE_EVENT_CANCELLED, "failed": SSE_EVENT_ERROR}
        sse_terminal = terminal_map[final_status]
        terminal_payload: dict[str, Any] = {"type": sse_terminal}
        if final_status == "failed":
            terminal_payload["message"] = error_reason or "internal_error"
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

    require_models_present(cfg)

    conversation_id = await get_or_create_conversation(list_id)

    conv_row = await get_conv_row(conversation_id)
    existing_summary = conv_row["summary"] if conv_row else None

    # Snapshot history before inserting new messages — the producer adds the
    # current user message explicitly via user_message_text.
    # Filter to status='done' here, not in get_recent_messages, so the UI
    # conversation endpoint still sees cancelled/failed rows for 已停止/重试.
    history_rows = await get_recent_messages(conversation_id, limit=100)
    history_rows = [r for r in history_rows if r["status"] == VISIBLE_MESSAGE_STATUS]
    history = []
    for r in history_rows:
        entry = {"role": r["role"], "content": r["content"]}
        raw_blocks = r["tool_blocks"]
        if raw_blocks:
            try:
                entry["tool_blocks"] = json.loads(raw_blocks)
            except json.JSONDecodeError:
                logger.exception("malformed tool_blocks JSON in message_id=%s", r["id"])
        history.append(entry)

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

    ui_lang = http_request.headers.get(UI_LANG_HEADER, "en")

    # Spawn producer
    task = asyncio.create_task(
        run_chat_turn(
            message_id=assistant_msg_id,
            conversation_id=conversation_id,
            user_message_text=request.message,
            history=history,
            summary=existing_summary,
            source_ids=source_ids,
            ui_lang=ui_lang,
            cfg=cfg,
            registry=run_registry,
            user_msg_id=user_msg_id,
        )
    )
    buf = run_registry.register(assistant_msg_id, task)
    # Let client know the server-assigned id so Stop can target it before the
    # first delta arrives.  Appended to the buffer before the consumer starts
    # reading, so it is always the first event delivered.
    buf.append({"type": SSE_EVENT_META, "message_id": assistant_msg_id})

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


_REFERENCES_HEADER = {"en": "## References", "zh": "## 引用来源"}


async def _load_sources_for_message(message: dict) -> dict[str, Any]:
    """{source_id: row} for every source cited in message.metadata.content_blocks.
    Empty/missing content_blocks → {}.
    """
    ids = {
        b.get("source_id")
        for b in (message.get("metadata") or {}).get("content_blocks", [])
        if b.get("type") == "citation" and b.get("source_id")
    }
    return await get_sources_by_ids(list(ids))


def _build_chat_message_markdown(
    message: dict,
    sources_by_id: dict[str, Any],
    *,
    lang: str = "en",
) -> str:
    """Verbatim prose + a localized References section listing each [N] → source title @ timestamp.

    Citations whose `source_id` is not in sources_by_id are dropped from References
    (prose keeps the [N] marker). Citations with no timestamp get no `@ MM:SS` suffix.
    """
    prose = message["content"]
    citations = [b for b in (message.get("metadata") or {}).get("content_blocks", []) if b.get("type") == "citation"]
    if not citations:
        return prose

    seen: set[int] = set()
    lines: list[str] = []
    for c in citations:
        idx = c["index"]
        if idx in seen:
            continue
        seen.add(idx)
        source = sources_by_id.get(c["source_id"])
        if source is None:
            continue
        ts = c.get("timestamp_start")
        ts_str = f" @ {format_mmss(ts)}" if ts is not None else ""
        lines.append(f"[{idx}] {source['title']}{ts_str}")

    header = _REFERENCES_HEADER.get(lang, _REFERENCES_HEADER["en"])
    # Blank-line separators so plain react-markdown (no remark-gfm) renders each
    # reference on its own paragraph instead of collapsing \n into a soft break.
    return prose + f"\n\n{header}\n\n" + "\n\n".join(lines)


@router.post("/lists/{list_id}/chat/save-message", status_code=201)
async def save_chat_message_to_artifact(
    list_id: str,
    req: ChatSaveMessageRequest,
    http_request: Request,
) -> ArtifactResponse:
    """Save a finished assistant message as a markdown artifact.

    Verbatim prose + a References section listing each [N] → source title
    @ timestamp. No LLM, no job queue — direct write.
    """
    list_row = await get_list(list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    msg_row = await get_message(req.message_id)
    if msg_row is None or not await assert_message_in_list(req.message_id, list_id):
        raise HTTPException(status_code=404, detail="Message not found")
    if msg_row["role"] != "assistant":
        raise HTTPException(status_code=422, detail="Only assistant messages can be saved")
    if msg_row["status"] != VISIBLE_MESSAGE_STATUS:
        raise HTTPException(status_code=422, detail=f"Message must be done (status={msg_row['status']})")

    msg = dict(msg_row)
    raw_meta = msg.get("metadata")
    if isinstance(raw_meta, str) and raw_meta:
        msg["metadata"] = json.loads(raw_meta)

    sources_by_id = await _load_sources_for_message(msg)
    ui_lang = http_request.headers.get(UI_LANG_HEADER, "en")
    content = _build_chat_message_markdown(msg, sources_by_id, lang=ui_lang)

    user_prompt = (await get_user_prompt_for_assistant(req.message_id)) or ""
    name = user_prompt.strip() or next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    name = name[:60] + ("…" if len(name) > 60 else "")

    artifact_id = str(uuid4())
    content_path = bibilab_home() / "artifacts" / list_id / f"{artifact_id}.md"
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(content, encoding="utf-8")

    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name=name,
        type="chat_message",
        prompt=user_prompt,
        source_ids=list(sources_by_id.keys()),
        status="completed",
        content_path=str(content_path.relative_to(bibilab_home())),
    )
    saved = await get_artifact(artifact_id)
    return ArtifactResponse.from_row(dict(saved))


@debug_router.get("/debug/messages/{message_id}")
async def get_debug_dump(message_id: str):
    path = bibilab_home() / "debug" / f"{message_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Debug dump not found")
    return Response(content=path.read_bytes(), media_type="application/json")
