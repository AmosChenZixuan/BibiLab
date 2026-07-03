"""Stateless JSON chat endpoint for driving the RAG pipeline from an eval
framework — no persistence, no SSE, full retrieval telemetry (not narrowed to
only what the LLM cited). Intentionally does NOT reuse the SPA-shaped
rag/citations ledger `run_chat_turn` persists: that shape drops uncited
sections and truncates evidence to a first-chunk preview — both hide what a
grader needs to score retrieval and generation separately."""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from bibilab.config import AIConfig, BibilabConfig, deep_merge, get_config
from bibilab.db import get_list, get_sources_for_list
from bibilab.models.eval import (
    EvalChatRequest,
    EvalChatResponse,
    EvalFindPassagesCall,
    EvalLLMOverride,
    EvalReadSectionCall,
    EvalSection,
)
from bibilab.pipeline._shared import _classify_llm_error
from bibilab.pipeline.chat_tools import (
    FIND_PASSAGES_TOOL,
    READ_SECTION_TOOL,
    CitationRegistryEntry,
    execute_tool,
)
from bibilab.routers._model_gate import require_models_present
from bibilab.routers.chat import (
    SSE_EVENT_CITATION,
    SSE_EVENT_DELTA,
    SSE_EVENT_ERROR,
    SSE_EVENT_TOOL_CALL_START,
    SSE_EVENT_TOOL_RESULT,
    build_grounding_prompt,
    stream_with_tools,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _merge_ai_config(base: AIConfig, override: EvalLLMOverride | None) -> AIConfig:
    """Field-level merge: an omitted override field inherits `base`'s value.
    context_window / output_language are never overridable — the request only
    exposes the fields an eval framework needs to A/B-test."""
    if override is None:
        return base
    try:
        return AIConfig(**deep_merge(base.model_dump(), override.model_dump(exclude_none=True)))
    except ValidationError as e:
        # Cross-field constraint violated (e.g. max_output_tokens >= the
        # non-overridable context_window): a caller input problem, not a
        # server fault — surface it as 422, not an unclassified 500.
        # Messages only, never str(e): pydantic's full rendering embeds the
        # merged input dict, which contains the backend's api_key.
        raise HTTPException(status_code=422, detail="; ".join(err["msg"] for err in e.errors())) from e


def _build_find_passages_call(result: dict, registry: dict[str, CitationRegistryEntry]) -> EvalFindPassagesCall:
    sections = []
    for s in result.get("section_coverage", []):
        entry = registry.get(s["section_id"])
        if entry is None:
            continue
        sections.append(
            EvalSection(
                index=entry.index,
                section_id=str(entry.section_id),
                source_id=entry.source_id,
                source_title=entry.title,
                timestamp_start=entry.timestamp_start,
                timestamp_end=entry.timestamp_end,
                rerank_score=entry.rerank_score,
                full_text=entry.full_text or "",
                # Patched after the stream ends, once every citation event is in.
                cited=False,
            )
        )
    return EvalFindPassagesCall(
        query=result.get("query", ""),
        sections=sections,
        candidates_evaluated=result.get("candidates_evaluated"),
        sources_with_hits=result.get("sources_with_hits"),
        sources_total=result.get("sources_total"),
        reranked=result.get("reranked"),
        scoped_pool_size=result.get("scoped_pool_size"),
        facet_scope=result.get("facet_scope"),
    )


def _build_read_section_call(result: dict, registry: dict[str, CitationRegistryEntry]) -> EvalReadSectionCall | None:
    if not result.get("source_id"):
        # Resolution error (bad/unknown index) — nothing was read, no row.
        return None
    entry = registry.get(result.get("section_id", ""))
    if entry is None:
        return None
    return EvalReadSectionCall(
        index=entry.index,
        section_id=str(entry.section_id),
        source_id=entry.source_id,
        source_title=entry.title,
        full_text=entry.full_text or "",
        cited=False,
    )


@router.post("/eval/run_chat")
async def run_chat_eval(
    request: EvalChatRequest,
    cfg: BibilabConfig = Depends(get_config),
) -> EvalChatResponse:
    list_row = await get_list(request.list_id)
    if list_row is None:
        raise HTTPException(status_code=404, detail="List not found")

    require_models_present(cfg)

    source_rows = await get_sources_for_list(request.list_id)
    source_ids = [row["id"] for row in source_rows]

    effective_ai = _merge_ai_config(cfg.ai, request.llm)
    response_language = request.language or "en"
    system_message = build_grounding_prompt(response_language=response_language)

    citation_registry: dict[str, CitationRegistryEntry] = {}

    async def execute_tool_bound(name: str, args: dict, **kwargs) -> dict:
        return await execute_tool(tool_name=name, arguments=args, source_ids=source_ids, cfg=cfg, **kwargs)

    tools = [FIND_PASSAGES_TOOL, READ_SECTION_TOOL]
    messages = [{"role": "user", "content": request.query}]

    answer_parts: list[str] = []
    cited_indices: set[int] = set()
    tool_calls: list[EvalFindPassagesCall | EvalReadSectionCall] = []
    stats: dict = {}

    start = time.monotonic()
    try:
        async for event in stream_with_tools(
            messages=messages,
            cfg=effective_ai,
            tools=tools,
            execute_tool_fn=execute_tool_bound,
            system=system_message,
            registry=citation_registry,
            response_language=response_language,
            stats=stats,
        ):
            if event.type == SSE_EVENT_DELTA and event.content:
                answer_parts.append(event.content)
            elif event.type == SSE_EVENT_CITATION:
                data = json.loads(event.content)
                cited_indices.add(data["index"])
                answer_parts.append(f"[{data['index']}]")
            elif event.type == SSE_EVENT_TOOL_CALL_START:
                # Production (run_chat_turn) inserts a paragraph break when a
                # tool call interrupts the stream; without it the preamble and
                # the post-tool synthesis fuse into one run-on string (no
                # whitespace at all in CJK) that no real client ever renders.
                if answer_parts and answer_parts[-1] != "\n\n":
                    answer_parts.append("\n\n")
            elif event.type == SSE_EVENT_TOOL_RESULT:
                # Build the row NOW, not after the stream: full_text on the
                # shared registry is last-writer-wins, so a later call that
                # re-surfaces the same section would rewrite the evidence
                # this call actually showed the LLM.
                parsed = json.loads(event.content)
                call: EvalFindPassagesCall | EvalReadSectionCall | None = None
                if parsed["name"] == FIND_PASSAGES_TOOL.name:
                    call = _build_find_passages_call(parsed["result"], citation_registry)
                elif parsed["name"] == READ_SECTION_TOOL.name:
                    call = _build_read_section_call(parsed["result"], citation_registry)
                if call is not None:
                    tool_calls.append(call)
            elif event.type == SSE_EVENT_ERROR:
                # A tool call failed inside stream_with_tools. Production
                # records this turn as "tool_error" — keep the codes aligned
                # so an eval harness can tell a retrieval failure (or a
                # model-emitted malformed tool call) from a backend bug.
                raise HTTPException(status_code=500, detail={"error": "tool_error"})
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — classified below, mirrors run_chat_turn's producer catch
        logger.exception("eval_run_chat pipeline failed list_id=%s", request.list_id)
        raise HTTPException(status_code=500, detail={"error": _classify_llm_error(e)}) from e

    latency_ms = int((time.monotonic() - start) * 1000)
    answer = "".join(answer_parts)

    # cited comes from the citation events — the pipeline's authoritative
    # judgment of what rendered as a citation — never from re-parsing [N] out
    # of the answer text, where a marker the parser rejected (hallucinated or
    # non-citable index, passed through as plain text) would false-positive.
    for call in tool_calls:
        if isinstance(call, EvalFindPassagesCall):
            for section in call.sections:
                section.cited = section.index in cited_indices
        else:
            call.cited = call.index in cited_indices

    return EvalChatResponse(
        answer=answer,
        tool_calls=tool_calls,
        iterations_used=stats.get("iterations", 0),
        synthesis_forced=stats.get("synthesis_forced", False),
        latency_ms=latency_ms,
    )
