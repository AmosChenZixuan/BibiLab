from __future__ import annotations

import asyncio
import json
import time
import uuid

from bibilab.routers.chat import stream_with_tools, build_grounding_prompt
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.chat_tools import (
    RETRIEVE_TOOL,
    SURVEY_TOOL,
    RETRIEVE_SCOPED_TOOL,
    RETRIEVE_TOOL_NAMES,
    execute_tool,
    CitationRegistryEntry,
    build_tool_block_entry,
)
from bibilab.pipeline.citation_parser import parse_delta
from bibilab.config import AIConfig, BibilabConfig, load_config

from eval._utils import now_iso
from eval.models import EvalCase, EvalRun, RunCaseResult
from eval.storage import load_eval_set, save_eval_run

CHAT_TOOLS: list[ToolDefinition] = [
    RETRIEVE_TOOL,
    SURVEY_TOOL,
    RETRIEVE_SCOPED_TOOL,
]

CHAT_MAX_TOKENS = 4096


async def run_single_case(
    case: EvalCase,
    list_id: str,
    source_ids: list[str],
    source_map: dict[str, str],
    ai_cfg: AIConfig,
    backend_cfg: BibilabConfig,
    system_prompt: str,
) -> RunCaseResult:
    start = time.monotonic()
    try:
        registry: dict[str, CitationRegistryEntry] = {}
        tool_block_sink: list[dict] = []

        async def execute_tool_fn(name, args, registry=registry, source_map=source_map):
            result = await execute_tool(
                name, args, list_id=list_id, source_ids=source_ids,
                ui_lang="ui", cfg=backend_cfg,
                registry=registry, source_map=source_map,
            )

            if name in RETRIEVE_TOOL_NAMES:
                raw_chunks = result.get("_raw_chunks")
                entry = build_tool_block_entry(
                    tool_use_id=f"tool_{uuid.uuid4().hex[:8]}",
                    name=name,
                    arguments=args,
                    result=result,
                    raw_chunks=raw_chunks,
                )
                tool_block_sink.append(entry)

            return result

        messages = [{"role": "user", "content": case.question}]

        text_deltas: list[str] = []
        citations: list[dict] = []
        parse_buffer = ""

        # LLM timing: sum gaps between consecutive events, excluding tool execution gaps
        # (tool_call_start → tool_result). This isolates actual model inference time.
        llm_total_s = 0.0
        prev_ts = time.monotonic()
        in_tool_exec = False

        async for event in stream_with_tools(
            messages=messages,
            cfg=ai_cfg,
            tools=CHAT_TOOLS,
            execute_tool_fn=execute_tool_fn,
            system=system_prompt,
            llm_max_tokens=CHAT_MAX_TOKENS,
            registry=registry,
            source_map=source_map,
        ):
            now = time.monotonic()

            if not in_tool_exec:
                llm_total_s += now - prev_ts
            prev_ts = now

            if event.type == "tool_call_start":
                in_tool_exec = True
            elif event.type == "tool_result":
                in_tool_exec = False

            if event.type == "delta":
                index_to_entry = {e.index: e for e in registry.values()}
                parsed_events, parse_buffer = parse_delta(
                    event.content or "", parse_buffer, index_to_entry,
                )
                for pe in parsed_events:
                    if pe.type == "delta":
                        text_deltas.append(pe.content or "")
                    elif pe.type == "citation":
                        try:
                            data = json.loads(pe.content or "{}")
                        except json.JSONDecodeError:
                            data = {}
                        citations.append({
                            "index": data.get("index", 0),
                            "source_id": data.get("source_id", ""),
                            "chunk_ids": data.get("chunk_ids", []),
                        })
            elif event.type == "done":
                break
            elif event.type == "error":
                raise RuntimeError(f"Stream error: {event.content}")

        answer = "".join(text_deltas)
        llm_duration_ms = int(llm_total_s * 1000)

        return RunCaseResult(
            case_id=case.id,
            answer=answer,
            citations=citations,
            rag_calls=[b.get("result", {}).get("summary", {}) for b in tool_block_sink],
            tool_blocks=tool_block_sink,
            llm_duration_ms=llm_duration_ms,
            error=None,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return RunCaseResult(
            case_id=case.id,
            answer="",
            citations=[],
            rag_calls=[],
            tool_blocks=[],
            llm_duration_ms=0,
            error=str(exc),
        )


async def run_eval(
    eval_set_id: str,
    ai_cfg: AIConfig | None = None,
    system_prompt: str | None = None,
) -> EvalRun:
    from bibilab.db import get_sources_for_list
    from eval.config import get_response_language

    if ai_cfg is None:
        backend_cfg = load_config()
        ai_cfg = backend_cfg.ai
    else:
        backend_cfg = load_config()

    if system_prompt is None:
        system_prompt = build_grounding_prompt(response_language=get_response_language())

    eval_set = load_eval_set(eval_set_id)
    locked = eval_set.locked_cases

    if not locked:
        raise ValueError("No locked cases in eval set. Review and lock cases first.")

    rows = await get_sources_for_list(eval_set.list_id)
    rows_dict = [dict(r) for r in rows]
    source_ids = [r["id"] for r in rows_dict]
    source_map = {r["video_id"]: r["id"] for r in rows_dict}

    run_id = str(uuid.uuid4())
    total = len(locked)

    async def _run_one(case: EvalCase) -> RunCaseResult:
        return await run_single_case(
            case=case,
            list_id=eval_set.list_id,
            source_ids=source_ids,
            source_map=source_map,
            ai_cfg=ai_cfg,
            backend_cfg=backend_cfg,
            system_prompt=system_prompt,
        )

    tasks = [asyncio.create_task(_run_one(c)) for c in locked]
    results: list[RunCaseResult] = []
    done_count = 0

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        done_count += 1
        if result.error:
            print(f"[{done_count}/{total}] {result.case_id[:8]} ✗ {result.error}", flush=True)
        else:
            print(f"[{done_count}/{total}] {result.case_id[:8]} ✓ {result.llm_duration_ms}ms", flush=True)

    run = EvalRun(
        id=run_id,
        eval_set_id=eval_set_id,
        test_profile={
            "model": ai_cfg.model,
            "protocol": ai_cfg.protocol,
            "base_url": ai_cfg.base_url,
        },
        timestamp=now_iso(),
        cases=results,
    )
    save_eval_run(run)
    return run
