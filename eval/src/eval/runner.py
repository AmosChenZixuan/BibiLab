from __future__ import annotations

import asyncio
import json
import time
import uuid

from bibilab.routers.chat import stream_with_tools, build_grounding_prompt, CHAT_MAX_TOKENS
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.chat_tools import (
    RETRIEVE_TOOL,
    SURVEY_TOOL,
    RETRIEVE_SCOPED_TOOL,
    RETRIEVE_TOOL_NAMES,
    QUERY_LIST_METADATA_TOOL,
    GENERATE_REPORT_TOOL,
    execute_tool,
    CitationRegistryEntry,
)
from bibilab.config import AIConfig, BibilabConfig, load_config

from eval._utils import now_iso
from eval.dashboard import TaskDashboard
from eval.models import EvalCase, EvalRun, ProfileSnapshot, RunCaseResult
from eval.storage import load_eval_set, save_eval_run

CHAT_TOOLS: list[ToolDefinition] = [
    RETRIEVE_TOOL,
    SURVEY_TOOL,
    RETRIEVE_SCOPED_TOOL,
    QUERY_LIST_METADATA_TOOL,
    GENERATE_REPORT_TOOL,
]


async def run_single_case(
    case: EvalCase,
    list_id: str,
    source_ids: list[str],
    source_map: dict[str, str],
    ai_cfg: AIConfig,
    backend_cfg: BibilabConfig,
    system_prompt: str,
    on_status=None,
) -> RunCaseResult:
    try:
        registry: dict[str, CitationRegistryEntry] = {}
        tool_block_sink: list[dict] = []

        async def execute_tool_fn(name, args, registry=registry, source_map=source_map):
            return await execute_tool(
                name, args, list_id=list_id, source_ids=source_ids,
                ui_lang="ui", cfg=backend_cfg,
                registry=registry, source_map=source_map,
            )

        messages = [{"role": "user", "content": case.question}]

        text_deltas: list[str] = []
        citations: list[dict] = []

        # LLM timing: sum gaps between consecutive events, excluding tool execution gaps
        # (tool_call_start → tool_result). This isolates actual model inference time.
        llm_total_s = 0.0
        prev_ts = time.monotonic()
        in_tool_exec = False
        first_delta_seen = False

        def _status(s: str):
            if on_status:
                on_status(s)

        _status("waiting LLM")

        async for event in stream_with_tools(
            messages=messages,
            cfg=ai_cfg,
            tools=CHAT_TOOLS,
            execute_tool_fn=execute_tool_fn,
            system=system_prompt,
            llm_max_tokens=CHAT_MAX_TOKENS,
            registry=registry,
            source_map=source_map,
            tool_block_sink=tool_block_sink,
        ):
            now = time.monotonic()

            if not in_tool_exec:
                llm_total_s += now - prev_ts
            prev_ts = now

            if event.type == "tool_call_start":
                in_tool_exec = True
                try:
                    tc = json.loads(event.content or "{}")
                    q = (tc.get("arguments") or {}).get("query", "")
                    _status(f"retrieving \"{q[:40]}\"")
                except json.JSONDecodeError:
                    _status("retrieving")
            elif event.type == "tool_result":
                in_tool_exec = False
                _status("synthesizing")

            if event.type == "delta":
                if not first_delta_seen and (event.content or "").strip():
                    first_delta_seen = True
                    _status("writing answer")
                text_deltas.append(event.content or "")
            elif event.type == "citation":
                try:
                    data = json.loads(event.content or "{}")
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
            rag_calls=[
                b["result"]["summary"]
                for b in tool_block_sink
                if b.get("name") in RETRIEVE_TOOL_NAMES
                and isinstance(b.get("result"), dict)
                and isinstance(b["result"].get("summary"), dict)
            ],
            tool_blocks=tool_block_sink,
            llm_duration_ms=llm_duration_ms,
            error=None,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return RunCaseResult(
            case_id=case.id,
            answer="",
            citations=[],
            rag_calls=[],
            tool_blocks=[],
            llm_duration_ms=0,
            error=str(exc),
        )


DEFAULT_CONCURRENCY = 4


async def run_eval(
    eval_set_id: str,
    ai_cfg: AIConfig | None = None,
    system_prompt: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
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

    task_rows = [(c.id, f"{c.category:<12} {c.id[:8]}") for c in locked]
    sem = asyncio.Semaphore(max(1, concurrency))
    with TaskDashboard("Runner", task_rows) as dash:

        async def _run_one(case: EvalCase) -> RunCaseResult:
            async with sem:
                dash.start(case.id, status="dispatched")
                result = await run_single_case(
                    case=case,
                    list_id=eval_set.list_id,
                    source_ids=source_ids,
                    source_map=source_map,
                    ai_cfg=ai_cfg,
                    backend_cfg=backend_cfg,
                    system_prompt=system_prompt,
                    on_status=lambda s, _id=case.id: dash.update(_id, s),
                )
                if result.error:
                    dash.done(case.id, ok=False, status="failed", error=result.error)
                else:
                    dash.done(case.id, ok=True, status="answered")
                return result

        tasks = [asyncio.create_task(_run_one(c)) for c in locked]
        results: list[RunCaseResult] = []
        for task in asyncio.as_completed(tasks):
            results.append(await task)

    run = EvalRun(
        id=run_id,
        eval_set_id=eval_set_id,
        test_profile=ProfileSnapshot(
            model=ai_cfg.model,
            protocol=ai_cfg.protocol,
            base_url=ai_cfg.base_url,
        ),
        timestamp=now_iso(),
        cases=results,
    )
    save_eval_run(run)
    return run
