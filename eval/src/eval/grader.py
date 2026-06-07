from __future__ import annotations

import asyncio
import json
import time

from bibilab.pipeline._shared import _call_llm
from bibilab.config import AIConfig

from eval._utils import now_iso, strip_json_fences
from eval.dashboard import TaskDashboard
from eval.models import GradeResult, GradedRun, ProfileSnapshot, RunCaseResult
from eval.storage import load_eval_set, load_eval_run, save_graded_run

RUBRIC = """Rating scale:
1 - Completely fails the dimension
2 - Mostly fails, minor success
3 - Mixed — some correct, some wrong
4 - Mostly correct, minor issues
5 - Perfect"""

_LANG_INSTRUCTION = {
    "en": "Write the `reasoning` field in English.",
    "zh": "Write the `reasoning` field in Chinese (用中文).",
}


def _lang_suffix(language: str) -> str:
    return "\n" + _LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION["zh"])


def build_context_relevance_prompt(question: str, chunks_text: str, language: str = "zh") -> str:
    return f"""Evaluate CONTEXT RELEVANCE: do the retrieved chunks contain the information needed to answer the question?

Question: {question}

Retrieved chunks:
{chunks_text}

{RUBRIC}

Return ONLY valid JSON. Do not add any explanation or markdown fences.
Format: {{"score": <1-5>, "reasoning": "<quote specific gaps or confirm coverage>"}}{_lang_suffix(language)}"""


def build_groundedness_prompt(answer: str, chunks_text: str, language: str = "zh") -> str:
    return f"""Evaluate GROUNDEDNESS: does every claim in the answer have support in the retrieved chunks?

Answer:
{answer}

Retrieved chunks:
{chunks_text}

{RUBRIC}

Return ONLY valid JSON. Do not add any explanation or markdown fences.
Format: {{"score": <1-5>, "reasoning": "<quote unsupported claims or confirm all claims sourced>"}}{_lang_suffix(language)}"""


def build_answer_relevance_prompt(question: str, answer: str, language: str = "zh") -> str:
    return f"""Evaluate ANSWER RELEVANCE: does the answer directly and completely address the question?

Question: {question}

Answer:
{answer}

{RUBRIC}

Return ONLY valid JSON. Do not add any explanation or markdown fences.
Format: {{"score": <1-5>, "reasoning": "<explain completeness and directness>"}}{_lang_suffix(language)}"""


def parse_grade_response(response: str) -> tuple[int | None, str]:
    raw = strip_json_fences(response)
    try:
        data = json.loads(raw)
        score = data.get("score")
        reasoning = data.get("reasoning", "")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return (None, f"Score not numeric: {score!r}")
        score_int = int(round(score))
        if score_int < 1 or score_int > 5:
            return (None, f"Score out of range: {score}")
        return (score_int, reasoning)
    except (json.JSONDecodeError, ValueError) as e:
        return (None, f"Failed to parse grade response: {e}")


def _chunks_text_from_case(case_result: RunCaseResult) -> str:
    # The exact tool messages the LLM read (find_passages bodies AND read_source
    # narratives), already carrying citation headers/fences. Grading must judge
    # against this — not a reconstruction from tool_blocks — or groundedness and
    # context-relevance measure a context the model never saw.
    if not case_result.llm_context:
        return "(no chunks retrieved)"
    return "\n\n".join(case_result.llm_context)


async def _grade_one(prompt: str, ai_cfg: AIConfig) -> tuple[int | None, str, int]:
    t0 = time.monotonic()
    try:
        raw = await asyncio.to_thread(_call_llm, prompt, ai_cfg, llm_timeout=120)
        score, reasoning = parse_grade_response(raw)
        if score is None:
            raw2 = await asyncio.to_thread(
                _call_llm,
                prompt + "\n\nYour previous response was invalid. Return ONLY valid JSON.",
                ai_cfg,
                llm_timeout=120,
            )
            score2, reasoning2 = parse_grade_response(raw2)
            llm_ms = int((time.monotonic() - t0) * 1000)
            if score2 is None:
                return (None, f"Retry also failed: {reasoning2}", llm_ms)
            return (score2, reasoning2, llm_ms)
        llm_ms = int((time.monotonic() - t0) * 1000)
        return (score, reasoning, llm_ms)
    except Exception as e:
        llm_ms = int((time.monotonic() - t0) * 1000)
        return (None, f"LLM call failed: {e}", llm_ms)


async def _grade_case(
    case_result: RunCaseResult,
    question: str,
    ai_cfg: AIConfig,
    language: str = "zh",
    on_dim_done=None,
) -> GradeResult:
    chunks_text = _chunks_text_from_case(case_result)
    answer = case_result.answer or "(no answer)"

    async def _wrap(prompt, dim):
        out = await _grade_one(prompt, ai_cfg)
        if on_dim_done:
            on_dim_done(dim, out[0] is not None)
        return out

    (cr_score, cr_reasoning, cr_ms), (g_score, g_reasoning, g_ms), (ar_score, ar_reasoning, ar_ms) = await asyncio.gather(
        _wrap(build_context_relevance_prompt(question, chunks_text, language), "CR"),
        _wrap(build_groundedness_prompt(answer, chunks_text, language), "G"),
        _wrap(build_answer_relevance_prompt(question, answer, language), "AR"),
    )

    return GradeResult(
        case_id=case_result.case_id,
        context_relevance=cr_score,
        context_relevance_reasoning=cr_reasoning or "Failed to grade",
        groundedness=g_score,
        groundedness_reasoning=g_reasoning or "Failed to grade",
        answer_relevance=ar_score,
        answer_relevance_reasoning=ar_reasoning or "Failed to grade",
        llm_duration_ms=cr_ms + g_ms + ar_ms,
    )


async def grade_run(
    run_id: str,
    ai_cfg: AIConfig | None = None,
    language: str = "zh",
) -> GradedRun:
    if ai_cfg is None:
        from bibilab.config import load_config

        ai_cfg = load_config().ai

    run = load_eval_run(run_id)
    eval_set = load_eval_set(run.eval_set_id)
    case_map = {c.id: c for c in eval_set.cases}

    task_rows = [(cr.case_id, f"{(case_map.get(cr.case_id).category if case_map.get(cr.case_id) else '?'):<12} {cr.case_id[:8]}") for cr in run.cases]

    with TaskDashboard("Grader", task_rows) as dash:
        async def _run_one(case_result: RunCaseResult) -> GradeResult:
            dash.start(case_result.case_id, status="CR ⠋  G ⠋  AR ⠋")
            dim_state = {"CR": "⠋", "G": "⠋", "AR": "⠋"}

            def _dim_done(dim: str, ok: bool):
                dim_state[dim] = "✓" if ok else "✗"
                dash.update(
                    case_result.case_id,
                    f"CR {dim_state['CR']}  G {dim_state['G']}  AR {dim_state['AR']}",
                )

            eval_case = case_map.get(case_result.case_id)
            question = eval_case.question if eval_case else case_result.case_id
            grade = await _grade_case(case_result, question, ai_cfg, language, on_dim_done=_dim_done)
            scores = f"CR={grade.context_relevance} G={grade.groundedness} AR={grade.answer_relevance}"
            ok = grade.context_relevance is not None and grade.groundedness is not None and grade.answer_relevance is not None
            dash.done(case_result.case_id, ok=ok, status=scores)
            return grade

        tasks = [asyncio.create_task(_run_one(cr)) for cr in run.cases]
        grades: list[GradeResult] = []
        for task in asyncio.as_completed(tasks):
            grades.append(await task)

    gr = GradedRun(
        run_id=run_id,
        grade_profile=ProfileSnapshot(
            model=ai_cfg.model,
            protocol=ai_cfg.protocol,
            base_url=ai_cfg.base_url,
        ),
        timestamp=now_iso(),
        grades=grades,
    )
    save_graded_run(gr)
    return gr
