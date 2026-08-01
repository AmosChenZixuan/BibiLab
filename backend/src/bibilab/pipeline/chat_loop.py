"""Bounded tool-calling loop and the prompt text that steers it.

Extracted from routers/chat.py, which stays the HTTP + SSE surface. Nothing
here touches an HTTP concept: `stream_with_tools` wraps `stream_llm` in a
bounded Plan → Act → Reflect loop, and the prompt builders assemble the
system prompt plus the tail-injected directives the loop appends.
"""

import json
import logging
from collections.abc import AsyncGenerator

from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _LANG_NATIVE_NAME,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    _no_text_error,
    stream_llm,
)
from bibilab.pipeline.chat_tools import (
    RETRIEVE_TOOL_NAMES,
    CitationRegistryEntry,
    build_tool_block_entry,
    strip_internal,
)
from bibilab.pipeline.citation_parser import flush_buffer, parse_delta

logger = logging.getLogger(__name__)


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

# Machine-readable error codes carried as SSE `error` content and persisted as
# the terminal ledger's error_reason; the frontend i18n-renders them. The eval
# endpoint imports ERROR_CODE_TOOL to keep the eval↔prod contract on one literal.
ERROR_CODE_TOOL = "tool_error"
ERROR_CODE_PERSISTENCE = "persistence_error"

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


def _native_lang_name(response_language: str) -> str:
    """Map a language code to its native display name for LLM directives, falling
    back to English for unknown codes — smaller models follow a readable name more
    reliably than a raw ISO code. Single source for the fallback so every
    tail-injected directive and the system prompt name the language identically."""
    return _LANG_NATIVE_NAME.get(response_language, "English")


def _build_preamble_trigger(response_language: str) -> str:
    """Build the preamble trigger with a trailing response-language clause.

    The trigger is injected at the message tail, making it the highest-recency
    instruction at the tool-call decision point — stronger than the system
    prompt's far-away `Respond in X` directive. Re-attached every non-synthesis turn,
    so it also anchors the post-tool *answer* turn — where source-language chunks (e.g.
    Chinese transcript) otherwise out-pull the distant system directive. The clause must
    govern the answer too: scoping it to "these sentences" left the answer anchorless
    once the model skips the preamble, dropping the clause with it.
    """
    return (
        f"{_PREAMBLE_TRIGGER} Write everything you output — both these preamble "
        f"sentences and your final answer — in {_native_lang_name(response_language)}, "
        f"regardless of the language of the retrieved source material."
    )


def _build_synthesis_directive(response_language: str) -> str:
    """Build the forced-synthesis directive with a trailing response-language
    clause. Like the preamble trigger, this is appended at the message tail (when
    tool iterations are exhausted) and out-competes the system prompt's language
    directive — and it produces the *final answer*, so a leaked-language answer
    here is worse than a leaked preamble. The clause keeps it in the answer's
    language."""
    return f"{_SYNTHESIS_DIRECTIVE} Respond in {_native_lang_name(response_language)}."


def _attach_preamble_trigger(messages: list[dict], protocol: str, response_language: str) -> list[dict]:
    """Return a copy of `messages` with the preamble trigger at the tail.

    Folds the trigger into the trailing user message when there is one (the initial
    question, or an Anthropic tool_result turn) so we never emit two consecutive
    user turns that a strict chat template might merge or drop. When the tail is not
    a user message (OpenAI tool messages), append a new user turn instead.
    """

    trigger = _build_preamble_trigger(response_language)
    msgs = list(messages)
    tail = msgs[-1] if msgs else None
    if not tail or tail.get("role") != "user":
        msgs.append({"role": "user", "content": trigger})
        return msgs

    content = tail["content"]
    if protocol == "anthropic":
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = list(content)
        else:
            raise TypeError(
                f"_attach_preamble_trigger: unexpected Anthropic user content type {type(content).__name__}"
            )
        blocks.append({"type": "text", "text": trigger})
        msgs[-1] = {"role": "user", "content": blocks}
    elif isinstance(content, str):
        msgs[-1] = {"role": "user", "content": f"{content}\n\n{trigger}"}
    else:
        # OpenAI multimodal/list content — append rather than risk mangling it.
        msgs.append({"role": "user", "content": trigger})
    return msgs


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
    lang = _native_lang_name(response_language)
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
    response_language: str = "en",
    stats: dict | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    if registry is None:
        registry = {}

    messages = list(messages)
    messages = _attach_preamble_trigger(messages, cfg.protocol, response_language)
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
                messages.append({"role": "user", "content": _build_synthesis_directive(response_language)})
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
                    # Machine code, not prose: the frontend i18n-renders this
                    # inline immediately; the terminal event repeats the code.
                    yield StreamEvent(type=SSE_EVENT_ERROR, content=ERROR_CODE_TOOL)
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
                anthropic_content = ([{"type": "text", "text": round_text}] if round_text.strip() else []) + [
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
                messages.append(
                    {
                        "role": "assistant",
                        "content": round_text if round_text.strip() else None,
                        "tool_calls": openai_tool_calls,
                    }
                )
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
                messages = _attach_preamble_trigger(messages, cfg.protocol, response_language)
            continue
    finally:
        # Export the cumulative LLM message list to the caller's sink (if provided).
        # try/finally guarantees the sink reflects the final state on every exit
        # path — normal return, early return from tool error, or exception.
        if messages_sink is not None:
            messages_sink.clear()
            messages_sink.extend(messages)
        if stats is not None:
            stats["iterations"] = iteration
            stats["synthesis_forced"] = synthesis_directive_sent
