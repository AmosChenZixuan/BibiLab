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

from bibilab.config import AIConfig, BibilabConfig, bibilab_home, get_config
from bibilab.db import (
    VISIBLE_MESSAGE_STATUS,
    assert_message_in_list,
    delete_conversation,
    get_conversation_by_list,
    get_list,
    get_or_create_conversation,
    get_recent_messages,
    get_sources_for_list,
    set_active_stream,
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
from bibilab.pipeline._shared import (
    _LANG_NATIVE_NAME,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    _classify_llm_error,
    _no_text_error,
    resolve_response_language,
    stream_llm,
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
    build_tool_block_entry,
    execute_tool,
    expand_message_for_provider,
    reseed_citation_registry,
    strip_internal,
)
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta
from bibilab.routers._model_gate import require_models_present

logger = logging.getLogger(__name__)

router = APIRouter()
debug_router = APIRouter()


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
# First event of every stream; carries {message_id} so the client can wire
# cancel-by-id before the first delta arrives (see web useSSEStream reattach path).
SSE_EVENT_META = "meta"

# All tools in v2 loop back (no terminal tool — the v1 `generate_report` was
# retired). Tool-call-start events + the LLM feed-back path therefore fire for
# every tool call when reached.
MAX_TOOL_ITERATIONS = 3

# Synthesis-turn directive: the tool budget is exhausted, so the model must
# answer in prose now. Tools stay *advertised* on this turn (see stream_with_tools)
# so the serving layer's tool-call grammar stays active — a stubborn tool attempt
# then parses as a structured (ignored) tool_call instead of leaking its native
# tool-call tokens as plain text into the answer.
_SYNTHESIS_DIRECTIVE = (
    "You have used all available tool calls. Do not call any more tools. "
    "Answer the user's question now in prose, using only the information already "
    "retrieved. If the retrieved content is insufficient, say so plainly."
)

_PREAMBLE_TRIGGER = (
    "[System directive — never confirm, restate, or acknowledge this to the user; just follow it silently.] "
    "Before EVERY tool call, your first output must be one or two short, natural sentences saying what you're "
    "after this step and why: how you framed it, or after a result, what it gave you and why you need another step. "
    "Speak plainly; never name the tools, their parameters, or index numbers. Then make the call in the same turn. "
    "Only when you already have enough to answer, skip this and answer directly."
)


def _attach_preamble_trigger(messages: list[dict], protocol: str) -> list[dict]:
    """Return a copy of `messages` with the trigger folded (Anthropic) or appended (other) at the tail."""

    msgs = list(messages)
    if protocol == "anthropic" and msgs and msgs[-1].get("role") == "user":
        content = msgs[-1]["content"]
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = list(content)
        else:
            raise TypeError(
                f"_attach_preamble_trigger: unexpected Anthropic user content type {type(content).__name__}"
            )
        blocks.append({"type": "text", "text": _PREAMBLE_TRIGGER})
        msgs[-1] = {"role": "user", "content": blocks}
    else:
        msgs.append({"role": "user", "content": _PREAMBLE_TRIGGER})
    return msgs


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


def _client_tool_result(result: dict) -> dict:
    """Strip internal fields (_-prefixed keys) before sending tool results to the client.

    Tool implementations may attach private metadata via _-prefixed keys (e.g. _chunks).
    These are never exposed over SSE. If you add a new tool whose result includes fields
    the client needs, do NOT prefix them with ``_``.
    """
    return strip_internal(result)


def _llm_tool_message_content(result: dict) -> str:
    """LLM-bound content of a tool result: the formatted excerpts only.

    `_chunks` is set on every tool path (find_passages, read_section narrative,
    resolution-error). Other fields (FTS bigram text, telemetry, bookkeeping)
    are client-only or persistence-only.
    """
    return result["_chunks"]


def build_grounding_prompt(response_language: str) -> str:
    """Build the system prompt for chat grounding.

    response_language is a language code (e.g. "en", "zh"), mapped to a
    human-readable display name for a single response-language directive
    placed at the tail — strongest recency, no repetition. The tail
    directive governs all output, including no-content refusals.
    """
    lang = _LANG_NATIVE_NAME.get(response_language, "English")
    return (
        "## Workflow\n"
        "You answer questions about a collection of video transcripts using two tools, both at "
        "SECTION granularity (a source is split into bounded sections, each with its own [N] "
        "citation index).\n\n"
        "- `find_passages(query, sequence_number?, season_number?)`: search for relevant excerpts. "
        "Your DEFAULT locator. Returns passages GROUPED BY SECTION — each section fenced under a [N] "
        "index with its summary, matching excerpts quoted beneath. Pass sequence_number / "
        "season_number ONLY when the message explicitly names an episode (第八集) or season (第二季); "
        "a facet match instead returns that episode's full section OUTLINE (every section's summary, "
        "one [N] each, no excerpts) for orientation — verbatim text comes only from a section's own [N].\n"
        "- `read_section(section_id)`: read ONE section's full verbatim transcript by its [N]. Use it "
        "to escalate when find_passages shows a section is on-topic but its fragments miss the asked "
        "specific; to read a whole episode verbatim, issue parallel read_section calls, one per [N].\n\n"
        "Query phrasing (the `query` argument to find_passages): write it as a natural-language "
        "question or noun phrase built around the DISTINCTIVE subject — the proper noun, name, or "
        "specific topic asked about. Do NOT flatten it into space-separated keywords, and do NOT append "
        "generic intent words (做法/步骤/方法/教程/介绍/原理, 'how to', 'guide', 'overview'): those recur in "
        "almost every transcript, so they match everything and bury the subject. E.g. write "
        "`光合作用是怎么进行的` not `光合作用 过程 原理 步骤`; write `量子计算` not `量子计算 应用 介绍 讲解`.\n\n"
        "Work as an agent in up to three Plan → Act → Reflect rounds:\n"
        "- PLAN: break the message into distinct information NEEDS (each entity, episode, or compared "
        "item is one need); classify each with the playbook below.\n"
        "- ACT: issue the planned calls. Independent needs → parallel calls in ONE round (one per need, "
        "the right tool each). A need that depends on a prior result → a sequential call next round.\n"
        "- REFLECT (after each result, per need): fragments or outline answer it → synthesize and stop; "
        "section on-topic but fragments miss the specific → read_section that [N] once, then answer; "
        "off-topic or corpus clearly lacks it → say the library has no content on it and stop.\n\n"
        "Playbook (need shape → strategy):\n"
        "- Single fact / definition / yes-no → 1× find_passages in natural language; "
        "missing specific → read_section once.\n"
        "- Locate (which episode / where) → 1× find_passages, no facet; answer with the [N] / timestamp.\n"
        "- Multiple independent subjects → parallel find_passages, one per subject.\n"
        "- Comparison across episodes → parallel find_passages, one per episode, EACH with its own sequence_number.\n"
        "- Multi-hop (one answer feeds the next query) → sequential find_passages; each hop uses the prior result.\n"
        "- Coverage (第N集讲了什么 / what episode N covers) → find_passages with the episode facet, then "
        "synthesize from the section OUTLINE summaries; read_section only to quote a specific "
        "section; do NOT re-search.\n"
        "- Enumeration (有哪些 / list them) → locate the sections, then read_section in full "
        "(top-k fragments miss scattered items).\n"
        "- Why / causal / possibly-absent → 1× find_passages; if the fragments hold no cause, the corpus "
        "likely lacks it — say so; do NOT reword-and-retry.\n"
        "- Follow-up answerable from history → answer directly, no tool.\n"
        "- Out of scope (opinion / real-world / speculation) → no tool; say the library does not cover it.\n\n"
        "Stopping discipline:\n"
        "- ONE retrieval per need. Reformulating the SAME need with different words is not allowed — "
        "escalate with read_section (on-topic, missing specific) or abstain (corpus lacks it). A multi-hop "
        "hop is a NEW need derived from the prior result, not a re-search of the same need.\n"
        "- Keep the SAME sequence_number / season_number across every call about the same episode this turn.\n"
        "- After read_section, answer from it — do not re-search the same episode.\n\n"
        "Trivial messages (greetings, thanks, capability questions, pure acknowledgments like '嗯', 'ok', "
        "'我懂了') get a direct reply WITHOUT calling any tool. A coverage question is NEVER answerable "
        "from conversation history — NEVER answer a coverage question from history; always retrieve the "
        "outline. Otherwise, if the current question is already answerable from the CONVERSATION HISTORY "
        "(you answered it, or a closely related question, earlier), answer directly from it without "
        "calling a tool.\n\n"
        "## Grounding\n"
        "Build your answer from retrieved excerpts / read sections alone. Do not draw on "
        "outside knowledge. Treat the content as authoritative whether fictional or real; "
        "never refuse on the grounds that content is fictional, informal, or not encyclopedic. "
        "Copy proper nouns (titles, names, terms) verbatim from the section they appear in. "
        'Each find_passages excerpt is fenced under its section by a `===== [N] "title" · '
        "Section S =====` line; never carry a proper noun across a fence. If find_passages "
        "returns no excerpts, tell the user that the library has no content on this topic, "
        "and stop — do not use outside knowledge, real-world analogies, or encyclopedic "
        "definitions. If a scoped search (sequence_number / season_number) matched no source, "
        "say so before answering from the wider pool. If read_section reports a section has "
        "no transcript available, you cannot answer from it — tell the user it is not "
        "available yet and do NOT infer from its title, summary, or duration.\n\n"
        "## Citation\n"
        "Cite each claim with `[N]`, where N is the section index from the tool result. "
        "Cite `[N]` ONLY for sections whose verbatim you were shown — either a find_passages "
        "excerpt under that [N], or a read_section call on that [N]. Outline summaries "
        "(the per-section [N] entries returned by a facet-matched find_passages) are "
        "orientation, not evidence: do not attach `[N]` to a claim drawn ONLY from a "
        "summary. Place `[N]` immediately after the sentence it supports, on the same line. "
        'For read_section answers, reference moments inline, e.g. "around 1:52 [1]".\n\n'
        "## Style\n"
        "Be direct and concise. Do not ask follow-up questions or offer unsolicited next steps. "
        f"Respond in {lang}."
    )


async def stream_with_tools(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition],
    execute_tool_fn,
    system: str | None = None,
    registry: dict[str, CitationRegistryEntry] | None = None,
    tool_block_sink: list[dict] | None = None,
    messages_sink: list[dict] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    if registry is None:
        registry = {}

    messages = list(messages)
    messages = _attach_preamble_trigger(messages, cfg.protocol)
    seen_chunk_ids: set[str] = set()
    iteration = 0
    parse_buffer = ""
    citation_emitted = False
    tool_used = False
    text_generated = False
    synthesis_directive_sent = False
    error_yielded = False
    last_stop_reason: str | None = None

    def _build_lookup() -> dict[int, CitationRegistryEntry]:
        return {e.index: e for e in registry.values()}

    async def _execute_with_registry(name: str, args: dict) -> dict:
        return await execute_tool_fn(name, args, registry=registry, seen_chunk_ids=seen_chunk_ids)

    try:
        while True:
            iteration += 1
            tool_calls: list[ToolCall] = []
            round_text = ""  # text emitted in this round; goes into the assistant message
            lookup = _build_lookup()
            is_synthesis_turn = iteration > MAX_TOOL_ITERATIONS
            if is_synthesis_turn and not synthesis_directive_sent:
                # Tell the model the budget is spent so it answers in prose. Tools
                # stay advertised below (grammar on) — see _SYNTHESIS_DIRECTIVE.
                messages.append({"role": "user", "content": _SYNTHESIS_DIRECTIVE})
                synthesis_directive_sent = True
            # Keep tools advertised even on the synthesis turn: with tools in the
            # request the serving layer keeps its tool-call grammar active, so a
            # stubborn tool attempt parses as a structured tool_call (ignored below
            # via the is_synthesis_turn branch) instead of leaking native tool-call
            # tokens as the answer. Execution is gated, not advertisement.
            async for event in stream_llm(messages, cfg, list(tools), system=system):
                if event.type == "tool_call":
                    tool_calls.append(event.tool_call)
                elif event.type == "delta" and event.content:
                    text_generated = True
                    round_text += event.content
                    # Parse incrementally so citations and text reach the client as
                    # they arrive rather than waiting for the full LLM response.
                    parsed_events, parse_buffer = parse_delta(event.content, parse_buffer, lookup)
                    for pe in parsed_events:
                        if pe.type == "citation":
                            citation_emitted = True
                        yield pe
                elif event.type == "done":
                    last_stop_reason = event.stop_reason
                elif event.type == "delta":
                    pass
                else:
                    yield event

            if not tool_calls or is_synthesis_turn:
                for pe in flush_buffer(parse_buffer):
                    yield pe
                # If tools were used but no answer text was ever generated, force one
                # more LLM call so the user always gets a text response. Tools stay
                # advertised here too (grammar on) for the same anti-leak reason as
                # the synthesis turn — a tool attempt parses as a structured (ignored)
                # tool_call, never leaking native tokens as the answer.
                if not text_generated and tool_used:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have retrieved information from the sources. "
                                "Now answer the user's original question. Do not call "
                                "any tools. Provide a complete answer based solely on "
                                "the retrieved content."
                            ),
                        }
                    )
                    async for event in stream_llm(messages, cfg, list(tools), system=system):
                        if event.type == "delta" and event.content:
                            text_generated = True
                            parsed_events, parse_buffer = parse_delta(event.content, parse_buffer, lookup)
                            for pe in parsed_events:
                                yield pe
                        elif event.type == "done":
                            last_stop_reason = event.stop_reason
                        elif event.type in ("delta", "tool_call"):
                            # Drop a stubborn tool attempt: no answer this turn → the
                            # no-text error below fires. Never executed, never yielded.
                            pass
                        else:
                            # Forward error/other events so producer-side error handling
                            # can capture failures during forced synthesis.
                            if event.type == "error":
                                error_yielded = True
                            yield event
                    for pe in flush_buffer(parse_buffer):
                        yield pe
                # If the LLM produced no visible text across the whole turn
                # (first turn with no tools, tool-using turn where the model
                # never produced text, or forced synthesis also empty), surface
                # it as a typed error so _classify_llm_error maps to a code. Branch on
                # the terminal stop_reason: a length cutoff → llm_output_budget_exceeded
                # ("raise max output tokens"); anything else → llm_empty_response, so
                # we never give false budget advice for a refusal or transient blank.
                # Without this raise an empty assistant message would persist silently.
                # Skip if an error event was already yielded — that's the real cause.
                if not text_generated and not error_yielded:
                    raise _no_text_error(last_stop_reason, cfg.max_output_tokens)
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

            # All tools in v2 loop back; feed results to the LLM for the next iteration.
            tool_used = True
            if cfg.protocol == "anthropic":
                anthropic_content = ([{"type": "text", "text": round_text}] if round_text else []) + [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in tool_calls
                ]
                messages.append({"role": "assistant", "content": anthropic_content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": _llm_tool_message_content(results[tc.id]),
                            }
                            for tc in tool_calls
                        ],
                    }
                )
            else:
                openai_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in tool_calls
                ]
                messages.append({"role": "assistant", "content": round_text or None, "tool_calls": openai_tool_calls})
                for tc in tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": _llm_tool_message_content(results[tc.id]),
                        }
                    )
            # Skip the trigger on the forced synthesis turn — it must answer in prose.
            if iteration < MAX_TOOL_ITERATIONS:
                messages = _attach_preamble_trigger(messages, cfg.protocol)
            continue
    finally:
        # Export the cumulative LLM message list to the caller's sink (if provided).
        # try/finally guarantees the sink reflects the final state on every exit
        # path — normal return, early return from tool error, or exception.
        if messages_sink is not None:
            messages_sink.clear()
            messages_sink.extend(messages)


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
    list_id: str,
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
            elif event.type == "tool_result":
                parsed = json.loads(event.content)
                if parsed["name"] in RETRIEVE_TOOL_NAMES:
                    result = parsed["result"]
                    # Store raw section_coverage for now; narrow by emitted citations in finally.
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
            elif event.type == "error":
                logger.error("stream_with_tools error: %s", event.content)
                error_reason = "tool_error"
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

            # Narrow section_coverage to only sections whose [N] actually appeared
            # in assistant text. content_blocks (type: "citation") is fully populated
            # at this point. Reconstruct context[] from the section-keyed citation
            # registry for each retrieve call.
            if retrieve_calls:
                emitted_indices = {cb["index"] for cb in content_blocks if cb.get("type") == "citation"}
                if emitted_indices:
                    emitted_section_ids = {
                        sid for sid, entry in citation_registry.items() if entry.index in emitted_indices
                    }
                else:
                    emitted_section_ids = set()
                for call in retrieve_calls:
                    # Narrow section_coverage only when citations were emitted.
                    if emitted_section_ids:
                        call["section_coverage"] = [
                            s for s in call["section_coverage"] if s.get("section_id") in emitted_section_ids
                        ]
                    # Always reconstruct context[] from citation registry.
                    # One entry per section in section_coverage (narrowed or full).
                    section_ids_in_call = {s["section_id"] for s in call["section_coverage"]}
                    context_entries = []
                    for sid in section_ids_in_call:
                        entry = citation_registry.get(sid)
                        if entry is not None:
                            context_entries.append(
                                {
                                    "section_id": sid,
                                    "section_seq": entry.seq,
                                    "chunk_id": entry.first_chunk_id or "",
                                    "citation_index": entry.index,
                                    "source_id": entry.source_id,
                                    "source_title": entry.title or "",
                                    "timestamp_start": entry.timestamp_start or 0.0,
                                    "timestamp_end": entry.timestamp_end or 0.0,
                                    "rerank_score": entry.rerank_score or 0.0,
                                    "preview": entry.preview or "",
                                }
                            )
                    call["context"] = context_entries

            for rs in read_section_calls:
                # registry is keyed by section_id; look up by section_id first, fall
                # back to source_id for legacy messages persisted before the section-granularity migration.
                section_id = rs.get("section_id", "")
                entry = citation_registry.get(section_id) if section_id else None
                if entry is None:
                    entry = citation_registry.get(rs["source_id"])
                if entry is not None and not rs.get("source_title"):
                    rs["source_title"] = entry.title or ""
                # read_section rows carry no chunk context — the read is bounded
                # verbatim transcript, not a fenced locator result. A synthetic entry
                # with zeroed fields would render as "0:00 / 0.00" in the frontend
                # ledger; an empty array lets the renderer branch on tool_name and
                # show a "read in full" affordance instead.
                rs["context"] = []
            all_calls = retrieve_calls + read_section_calls

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
                error_reason = "persistence_error"
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


@debug_router.get("/debug/messages/{message_id}")
async def get_debug_dump(message_id: str):
    path = bibilab_home() / "debug" / f"{message_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Debug dump not found")
    return Response(content=path.read_bytes(), media_type="application/json")
