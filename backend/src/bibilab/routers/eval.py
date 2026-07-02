"""Stateless JSON chat endpoint for driving the RAG pipeline from an eval
framework — no persistence, no SSE, full retrieval telemetry (not narrowed to
only what the LLM cited). See docs/specs or issue #581 for the design
rationale; this intentionally does NOT reuse the SPA-shaped rag/citations
ledger `run_chat_turn` persists."""

import json
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException

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
from bibilab.routers.chat import build_grounding_prompt, stream_with_tools

logger = logging.getLogger(__name__)

router = APIRouter()

_CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")


def _merge_ai_config(base: AIConfig, override: EvalLLMOverride | None) -> AIConfig:
    """Field-level merge: an omitted override field inherits `base`'s value.
    context_window / output_language are never overridable — the request only
    exposes the fields an eval framework needs to A/B-test."""
    if override is None:
        return base
    return AIConfig(**deep_merge(base.model_dump(), override.model_dump(exclude_none=True)))


def _build_find_passages_call(
    result: dict, registry: dict[str, CitationRegistryEntry], cited: set[int]
) -> EvalFindPassagesCall:
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
                cited=entry.index in cited,
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


def _build_read_section_call(
    result: dict, registry: dict[str, CitationRegistryEntry], cited: set[int]
) -> EvalReadSectionCall | None:
    source_id = result.get("source_id")
    if not source_id:
        # Resolution error (bad/unknown index) — nothing was read, no row.
        return None
    section_id = result.get("section_id", "")
    entry = registry.get(section_id)
    return EvalReadSectionCall(
        section_id=str(section_id),
        source_id=source_id,
        source_title=result.get("source_title", ""),
        full_text=(entry.full_text if entry else None) or "",
        cited=entry is not None and entry.index in cited,
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
    tool_results: list[dict] = []
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
            if event.type == "delta" and event.content:
                answer_parts.append(event.content)
            elif event.type == "citation":
                data = json.loads(event.content)
                answer_parts.append(f"[{data['index']}]")
            elif event.type == "tool_result":
                tool_results.append(json.loads(event.content))
            elif event.type == "error":
                # A tool call failed inside stream_with_tools (see its
                # SSE_EVENT_ERROR branch). Not an SDK exception type, so
                # _classify_llm_error below falls through to "internal_error" —
                # same code an unclassified pipeline exception gets.
                raise RuntimeError(event.content)
    except Exception as e:  # noqa: BLE001 — classified below, mirrors run_chat_turn's producer catch
        logger.exception("eval_run_chat pipeline failed list_id=%s", request.list_id)
        raise HTTPException(status_code=500, detail={"error": _classify_llm_error(e)}) from e

    latency_ms = int((time.monotonic() - start) * 1000)
    answer = "".join(answer_parts)
    cited_indices = {int(m) for m in _CITATION_INDEX_RE.findall(answer)}

    tool_calls: list[EvalFindPassagesCall | EvalReadSectionCall] = []
    for parsed in tool_results:
        name = parsed["name"]
        result = parsed["result"]
        if name == FIND_PASSAGES_TOOL.name:
            tool_calls.append(_build_find_passages_call(result, citation_registry, cited_indices))
        elif name == READ_SECTION_TOOL.name:
            call = _build_read_section_call(result, citation_registry, cited_indices)
            if call is not None:
                tool_calls.append(call)

    return EvalChatResponse(
        answer=answer,
        tool_calls=tool_calls,
        iterations_used=stats.get("iterations", 0),
        synthesis_forced=stats.get("synthesis_forced", False),
        detected_language=response_language,
        latency_ms=latency_ms,
    )
