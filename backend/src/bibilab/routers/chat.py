import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import anthropic
import openai
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
    FIND_PASSAGES_TOOL,
    READ_SOURCE_TOOL,
    RETRIEVE_TOOL_NAMES,
    CitationRegistryEntry,
    build_tool_block_entry,
    execute_tool,
    expand_message_for_provider,
    reseed_citation_registry,
)
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta
from bibilab.routers._model_gate import require_models_present

logger = logging.getLogger(__name__)

router = APIRouter()


def resolve_response_language(cfg: AIConfig, ui_lang: str) -> str:
    """Return the language string to use in chat responses.

    AIConfig.output_language wins when explicitly set; "ui" means follow
    the UI's X-UI-Lang header (passed in as ui_lang).
    """
    return ui_lang if cfg.output_language == "ui" else cfg.output_language


# SSE event types — used both as stream-internal event discriminators (stream_llm yield)
# and as the 'type' field in the SSE 'data:' JSON payload sent to the client.
SSE_EVENT_DELTA = "delta"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"
SSE_EVENT_TOOL_RESULT = "tool_result"
SSE_EVENT_TOOL_CALL_START = "tool_call_start"
SSE_EVENT_CITATION = "citation"
SSE_EVENT_CANCELLED = "cancelled"
# Final authoritative rag.calls (persisted shape, with context[]) emitted just
# before the terminal event so the client ledger matches post-refresh state
# without a manual reload.
SSE_EVENT_RAG = "rag"

# Sized for thinking-capable models with potentially long chat responses + tool turns.
CHAT_MAX_TOKENS = 16384

LOOPBACK_TOOLS = RETRIEVE_TOOL_NAMES | {READ_SOURCE_TOOL.name}
MAX_TOOL_ITERATIONS = 3


_ERROR_CODE_MAP: tuple[tuple[type[Exception], str], ...] = (
    (openai.APIConnectionError, "llm_connection_error"),
    (anthropic.APIConnectionError, "llm_connection_error"),
    (openai.AuthenticationError, "llm_auth_error"),
    (openai.PermissionDeniedError, "llm_auth_error"),
    (anthropic.AuthenticationError, "llm_auth_error"),
    (openai.RateLimitError, "llm_rate_limit_error"),
    (anthropic.RateLimitError, "llm_rate_limit_error"),
    (openai.APIError, "llm_api_error"),
    (anthropic.APIError, "llm_api_error"),
)


def classify_error(exception: Exception) -> str:
    """Map SDK exception types to a stable error code for i18n on the frontend."""
    for exc_type, code in _ERROR_CODE_MAP:
        if isinstance(exception, exc_type):
            return code
    return "internal_error"


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


def build_grounding_prompt(response_language: str) -> str:
    """Build the system prompt for chat grounding.

    response_language is interpolated into the language directive and the
    fallback sentence so refusals match the user's UI/config language.
    """
    return (
        f"Respond in {response_language}.\n\n"
        "## Workflow\n"
        "If the user asks to generate a report, summary, study guide, blog post, "
        "or custom report — where the answer is a structured document, not a "
        "chat reply — call `generate_report` immediately. Do not retrieve first; "
        "the report pipeline handles its own retrieval.\n\n"
        "For content questions, first decompose the user's message into distinct "
        "SUBJECTS (entities, episodes, seasons, items compared). ONE subject → "
        "call ONE retrieval tool; do NOT hedge by calling multiple variants on "
        "the same subject. MULTIPLE subjects ('A 和 B 的区别', '第一集 xxx 第三集 yyy', "
        "multi-entity questions) → call the appropriate tool ONCE PER SUBJECT "
        "in parallel, each scoped to its own subject.\n\n"
        "Tools per subject:\n"
        "- `retrieve(query)`: single-fact lookups, definitions, what/when/who/why. "
        "Pass the user's question in natural form.\n"
        "- `survey(query)`: list-summary / episode-wide recap for ONE umbrella "
        "subject (e.g. '有哪些面食做法'). Wider retrieval pool than retrieve. "
        "Pass the user's question in natural form. Multi-subject comparisons "
        "use parallel calls (one per subject, appropriate tool each), not a "
        "single survey.\n"
        "- `retrieve_scoped(query, sequence_number?, season_number?)`: use ONLY "
        "when the CURRENT message explicitly names an episode (第八集) or season "
        "(第二季); do not infer scope from prior turns.\n\n"
        "For questions about source counts, durations, or languages, call "
        "`query_list_metadata`.\n\n"
        "Earlier turns' retrievals appear only as a one-line tag (the prior "
        "query and which sources were used) — the excerpt text itself is not "
        "replayed. You cannot cite or quote a prior turn's excerpts. To "
        "answer from that content, call retrieve / survey / retrieve_scoped "
        "again this turn for fresh excerpts. Prior excerpts about an "
        "unrelated topic are not grounds to refuse: only a fresh retrieve "
        "result can establish whether the library covers the new topic.\n\n"
        'If the user sends a pure acknowledgment ("嗯", "ok", "thanks", '
        '"我懂了") with no new question, respond naturally without calling '
        "any retrieve tool.\n\n"
        "## Grounding\n"
        "Build your answer from the retrieved excerpts alone. Do not draw on "
        "outside knowledge. Treat the excerpts as authoritative whether the "
        "content is fictional or real; never refuse on the grounds that content "
        "is fictional, informal, indirect, or not in encyclopedic form. "
        "Excerpts may be narrative, character dialog, debate, or informal "
        "mention — paraphrase what the excerpts say about the topic. Do not "
        "suggest the user reformulate the question or consult outside sources; "
        "only the retrieve tool result itself can declare zero coverage. Copy "
        "proper nouns (titles, character names, technical terms) verbatim from "
        "the excerpts they appear in — do not paraphrase them or substitute "
        "terms from a different source. Each retrieved excerpt is fenced under "
        "its source by a `===== Source [N] =====` line; never carry a proper "
        "noun across a fence. Never substitute outside knowledge, real-world "
        "analogies, or encyclopedic definitions for missing evidence. If the "
        "retrieved excerpts do not contain the answer, the retrieve tool result "
        "itself will tell you how to respond; follow that instruction and "
        "stop.\n\n"
        "## Citation\n"
        "Cite each claim with `[N]`, where N is the source index from the "
        "retrieve result. Cite only sources you actually retrieved. Place `[N]` "
        "immediately after the sentence it supports, on the same line; do not "
        "put a citation on its own line. For long sources, include the relevant "
        'timestamp inline, e.g. "around 2:00 [1]". Use natural phrasing, not a '
        "structured format.\n\n"
        "## Style\n"
        f"Answer in {response_language}. Be direct and concise. Do not ask "
        "follow-up questions or offer unsolicited next steps."
    )


async def stream_with_tools(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition],
    execute_tool_fn,
    system: str | None = None,
    llm_max_tokens: int = CHAT_MAX_TOKENS,
    registry: dict[str, CitationRegistryEntry] | None = None,
    tool_block_sink: list[dict] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    if registry is None:
        registry = {}

    messages = list(messages)
    iteration = 0
    parse_buffer = ""
    citation_emitted = False
    tool_used = False
    text_generated = False

    def _build_lookup() -> dict[int, CitationRegistryEntry]:
        return {e.index: e for e in registry.values()}

    async def _execute_with_registry(name: str, args: dict) -> dict:
        return await execute_tool_fn(name, args, registry=registry)

    while True:
        iteration += 1
        tool_calls: list[ToolCall] = []
        lookup = _build_lookup()
        is_synthesis_turn = iteration > MAX_TOOL_ITERATIONS
        active_tools = [] if is_synthesis_turn else list(tools)
        async for event in stream_llm(messages, cfg, active_tools, system=system, llm_max_tokens=llm_max_tokens):
            if event.type == "tool_call":
                tool_calls.append(event.tool_call)
            elif event.type == "delta" and event.content:
                text_generated = True
                # Parse incrementally so citations and text reach the client as
                # they arrive rather than waiting for the full LLM response.
                parsed_events, parse_buffer = parse_delta(event.content, parse_buffer, lookup)
                for pe in parsed_events:
                    if pe.type == "citation":
                        citation_emitted = True
                    yield pe
            elif event.type == "done" and tool_calls:
                pass
            elif event.type in ("delta", "done"):
                pass
            else:
                yield event

        if not tool_calls or is_synthesis_turn:
            for pe in flush_buffer(parse_buffer):
                yield pe
            # If tools were used but no answer text was ever generated, force one
            # more LLM call with no tools so the user always gets a text response.
            if not text_generated and tool_used:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have retrieved information from the sources. "
                            "Now answer the user's original question. "
                            "Provide a complete answer based solely on the retrieved content."
                        ),
                    }
                )
                async for event in stream_llm(messages, cfg, [], system=system, llm_max_tokens=llm_max_tokens):
                    if event.type == "delta" and event.content:
                        parsed_events, parse_buffer = parse_delta(event.content, parse_buffer, lookup)
                        for pe in parsed_events:
                            yield pe
                    elif event.type in ("delta", "done"):
                        pass
                    else:
                        # Forward error/other events so producer-side error handling
                        # can capture failures during forced synthesis.
                        yield event
                for pe in flush_buffer(parse_buffer):
                    yield pe
            if not citation_emitted and registry:
                logger.info(
                    "citations_missing_after_retrieve registry_size=%d",
                    len(registry),
                )
            return

        # Reset: a partial [ left over from preamble text should not bleed into
        # iteration 2's citation parsing.
        parse_buffer = ""

        retrieve_calls = [tc for tc in tool_calls if tc.name in RETRIEVE_TOOL_NAMES]
        if len(retrieve_calls) > 1:
            logger.info(
                "parallel_retrieve count=%d names=%r queries=%r",
                len(retrieve_calls),
                [tc.name for tc in retrieve_calls],
                [str(tc.arguments.get("query", ""))[:80] for tc in retrieve_calls],
            )

        results: dict[str, dict] = {}
        for tc in tool_calls:
            if tc.name in LOOPBACK_TOOLS:
                yield StreamEvent(
                    type=SSE_EVENT_TOOL_CALL_START,
                    content=json.dumps({"id": tc.id, "name": tc.name, "arguments": tc.arguments}),
                )
        for tc in tool_calls:
            try:
                result = await _execute_with_registry(tc.name, tc.arguments)
            except Exception:
                logger.exception("tool_execution_failed tool=%s", tc.name)
                yield StreamEvent(type=SSE_EVENT_ERROR, content=f"Tool {tc.name} failed")
                return

            results[tc.id] = result
            if tool_block_sink is not None:
                try:
                    tool_block_sink.append(
                        build_tool_block_entry(
                            tool_use_id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                            result=result,
                            raw_chunks=result.get("_raw_chunks"),
                        )
                    )
                except Exception:
                    logger.exception("tool_block_sink_append_failed tool=%s", tc.name)
            yield StreamEvent(
                type=SSE_EVENT_TOOL_RESULT,
                content=json.dumps({"id": tc.id, "name": tc.name, "result": _client_tool_result(result)}),
            )

        if any(tc.name in LOOPBACK_TOOLS for tc in tool_calls):
            tool_used = True
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
    elif event.type == SSE_EVENT_TOOL_RESULT:
        parsed = json.loads(event.content)
        return {"type": SSE_EVENT_TOOL_RESULT, **parsed}
    elif event.type == SSE_EVENT_ERROR:
        return {"type": SSE_EVENT_ERROR, "message": event.content}
    return None


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
    list_id: str,
    user_message_text: str,
    history: list[dict],
    summary: str | None,
    source_ids: list[str],
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
                list_id=list_id,
                source_ids=source_ids,
                ui_lang=ui_lang,
                cfg=cfg,
                **kwargs,
            )

        tools = [FIND_PASSAGES_TOOL, READ_SOURCE_TOOL]

        tool_blocks: list[dict] = []

        async for event in stream_with_tools(
            messages=messages_for_llm,
            cfg=cfg.ai,
            tools=tools,
            execute_tool_fn=execute_tool_bound,
            system=system_message if system_message.strip() else None,
            llm_max_tokens=CHAT_MAX_TOKENS,
            registry=citation_registry,
            tool_block_sink=tool_blocks,
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
                if parsed["name"] in RETRIEVE_TOOL_NAMES:
                    result = parsed["result"]
                    # Store raw source_coverage for now; narrow by emitted citations in finally.
                    retrieve_calls.append(
                        {
                            "query": result.get("query", ""),
                            "tool_name": result.get("tool_name", parsed["name"]),
                            "candidates_evaluated": result.get("candidates_evaluated"),
                            "sources_with_hits": result.get("sources_with_hits"),
                            "sources_total": result.get("sources_total"),
                            "source_coverage": result.get("source_coverage", []),
                            "reranked": result.get("reranked", False),
                            "scoped_pool_size": result.get("scoped_pool_size"),
                            "facet_scope": result.get("facet_scope"),
                        }
                    )
            elif event.type == "error":
                logger.error("stream_with_tools error: %s", event.content)
                error_reason = "tool_error"
                final_status = "failed"
                return
    except asyncio.CancelledError:
        final_status = "cancelled"
        raise
    except Exception as e:
        logger.exception("producer failed message_id=%s", message_id)
        error_reason = classify_error(e)
        final_status = "failed"
    finally:
        try:
            if pending_text:
                _flush_pending_text(content_blocks, pending_text)
                pending_text = ""

            # Narrow source_coverage to only sources whose [N] actually appeared in assistant text.
            # content_blocks (type: "citation") is fully populated at this point.
            # Reconstruct context[] from citation_registry for each retrieve call.
            if retrieve_calls:
                emitted_indices = {cb["index"] for cb in content_blocks if cb.get("type") == "citation"}
                if emitted_indices:
                    emitted_source_ids = {
                        sid for sid, entry in citation_registry.items() if entry.index in emitted_indices
                    }
                else:
                    emitted_source_ids = set()
                for call in retrieve_calls:
                    # Narrow source_coverage only when citations were emitted.
                    if emitted_source_ids:
                        call["source_coverage"] = [
                            s for s in call["source_coverage"] if s.get("source_id") in emitted_source_ids
                        ]
                    # Always reconstruct context[] from citation registry.
                    # One entry per source in source_coverage (narrowed or full).
                    source_ids_in_call = {s["source_id"] for s in call["source_coverage"]}
                    context_entries = []
                    for sid in source_ids_in_call:
                        entry = citation_registry.get(sid)
                        if entry is not None:
                            context_entries.append(
                                {
                                    "chunk_id": entry.first_chunk_id or "",
                                    "citation_index": entry.index,
                                    "source_id": sid,
                                    "source_title": entry.title or "",
                                    "timestamp_start": entry.timestamp_start or 0.0,
                                    "timestamp_end": entry.timestamp_end or 0.0,
                                    "rerank_score": entry.rerank_score or 0.0,
                                    "preview": entry.preview or "",
                                }
                            )
                    call["context"] = context_entries
                    call["scoped_pool_size"] = call.get("scoped_pool_size")
                    call["facet_scope"] = call.get("facet_scope")

            meta: dict[str, Any] = {}

            if retrieve_calls:
                meta["rag"] = {"calls": retrieve_calls}
            if content_blocks:
                meta["content_blocks"] = content_blocks

            assistant_content = "".join(assistant_text_deltas)
            error_text = error_reason if error_reason else ("internal_error" if final_status == "failed" else None)
            await update_message_content(
                message_id,
                content=assistant_content,
                metadata=meta if meta else None,
                status=final_status,
                error=error_text,
                tool_blocks=tool_blocks if tool_blocks else None,
            )
        except Exception:
            logger.exception("producer finalize failed message_id=%s", message_id)
            final_status = "failed"
            if error_reason is None:
                error_reason = "persistence_error"

        try:
            await set_active_stream(conversation_id, None)
        except Exception:
            logger.exception(
                "producer clear active_stream failed message_id=%s — stale pointer may cause "
                "reattach to dead stream on next page load",
                message_id,
            )

        # Final authoritative ledger: persisted-shape calls (context[]
        # reconstructed) so the client matches post-refresh without reloading.
        if retrieve_calls:
            buf.append({"type": SSE_EVENT_RAG, "calls": retrieve_calls})

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
    history_rows = await get_recent_messages(conversation_id, limit=100)
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
