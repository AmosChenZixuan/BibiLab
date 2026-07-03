from __future__ import annotations

import asyncio
import uuid

from eval import api
from eval._utils import now_iso
from eval.config import get_response_language
from eval.dashboard import TaskDashboard
from eval.models import EvalCase, EvalRun, RunCaseResult
from eval.storage import load_eval_set, save_eval_run

# Telemetry fields copied off a find_passages tool call into rag_calls rows.
_RAG_CALL_FIELDS = (
    "query",
    "candidates_evaluated",
    "sources_with_hits",
    "sources_total",
    "reranked",
    "scoped_pool_size",
    "facet_scope",
)


def map_response(case_id: str, body: dict) -> RunCaseResult:
    """Map a /api/eval/run_chat response into a RunCaseResult.

    llm_context comes straight from the endpoint's `llm_context` — the exact
    LLM-bound tool message per call (fence headers, facet notes and all), so
    grading judges against what the LLM actually read, not a reconstruction.
    """
    citations: list[dict] = []
    rag_calls: list[dict] = []

    for call in body.get("tool_calls", []):
        if call.get("tool_name") == "find_passages":
            sections = call.get("sections", [])
            rag_calls.append({k: call.get(k) for k in _RAG_CALL_FIELDS})
            cited = [s for s in sections if s.get("cited")]
        else:  # read_section
            cited = [call] if call.get("cited") else []
        for s in cited:
            citations.append(
                {"index": s["index"], "source_id": s["source_id"], "section_id": s["section_id"]}
            )

    return RunCaseResult(
        case_id=case_id,
        answer=body.get("answer", ""),
        citations=citations,
        rag_calls=rag_calls,
        tool_blocks=body.get("tool_calls", []),
        llm_context=body.get("llm_context", []),
        # Whole-turn wall time including tool execution — LLM-only timing is
        # not recoverable over the sync JSON endpoint.
        llm_duration_ms=body.get("latency_ms", 0),
        error=None,
    )


async def run_single_case(
    case: EvalCase,
    list_id: str,
    llm: dict | None,
    language: str,
    on_status=None,
) -> RunCaseResult:
    if on_status:
        on_status("waiting backend")
    try:
        body = await api.run_chat(query=case.question, list_id=list_id, llm=llm, language=language)
        return map_response(case.id, body)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return RunCaseResult(case_id=case.id, answer="", error=str(exc))


DEFAULT_CONCURRENCY = 4


async def run_eval(
    eval_set_id: str,
    llm: dict | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> EvalRun:
    language = get_response_language()

    eval_set = load_eval_set(eval_set_id)
    locked = eval_set.locked_cases

    if not locked:
        raise ValueError("No locked cases in eval set. Review and lock cases first.")

    # Resolved up front: records what will serve the run, and fails fast when
    # the backend is unreachable (every case would fail anyway).
    test_profile = api.effective_profile(llm)

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
                    llm=llm,
                    language=language,
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
        test_profile=test_profile,
        timestamp=now_iso(),
        cases=results,
    )
    save_eval_run(run)
    return run
