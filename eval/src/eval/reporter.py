from __future__ import annotations

import json
from typing import Any

from eval.models import GradeResult, GradedRun
from eval.storage import load_eval_set


def aggregate_scores(
    grades: list[GradeResult], category_map: dict[str, str]
) -> dict[str, dict[str, float]]:
    by_cat: dict[str, dict[str, list[float]]] = {}

    for g in grades:
        cat = category_map.get(g.case_id, "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"context_relevance": [], "groundedness": [], "answer_relevance": []}
        if g.context_relevance > 0:
            by_cat[cat]["context_relevance"].append(float(g.context_relevance))
        if g.groundedness > 0:
            by_cat[cat]["groundedness"].append(float(g.groundedness))
        if g.answer_relevance > 0:
            by_cat[cat]["answer_relevance"].append(float(g.answer_relevance))

    result: dict[str, dict[str, float]] = {}
    all_cr: list[float] = []
    all_g: list[float] = []
    all_ar: list[float] = []

    for cat, scores in by_cat.items():
        cr_mean = _mean(scores["context_relevance"])
        g_mean = _mean(scores["groundedness"])
        ar_mean = _mean(scores["answer_relevance"])
        result[cat] = {
            "context_relevance": cr_mean,
            "groundedness": g_mean,
            "answer_relevance": ar_mean,
        }
        all_cr.extend(scores["context_relevance"])
        all_g.extend(scores["groundedness"])
        all_ar.extend(scores["answer_relevance"])

    result["overall"] = {
        "context_relevance": _mean(all_cr),
        "groundedness": _mean(all_g),
        "answer_relevance": _mean(all_ar),
    }
    return result


def diff_scores(
    current: dict[str, dict[str, float]], previous: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    diff: dict[str, dict[str, float]] = {}
    for cat in current:
        if cat in previous:
            diff[cat] = {
                k: round(current[cat][k] - previous[cat][k], 1) for k in current[cat]
            }
    return diff


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def format_report_text(
    eval_set_name: str,
    case_count: int,
    test_model: str,
    grade_model: str,
    aggregate: dict[str, dict[str, float]],
    diff: dict[str, dict[str, float]] | None = None,
    compare_model: str | None = None,
) -> str:
    lines = [
        f"Eval Set: {eval_set_name} ({case_count} cases)",
        f"Test: {test_model} | Grade: {grade_model}",
    ]
    if compare_model:
        lines.append(f"Compared: {compare_model}")

    lines.append("")
    lines.append("              Context     Groundedness  Answer")
    lines.append("              Relevance               Relevance")

    dims = ["context_relevance", "groundedness", "answer_relevance"]
    categories = [k for k in aggregate if k != "overall"]

    for cat in sorted(categories):
        scores = aggregate[cat]
        parts = [f"{cat:<12}"]
        for d in dims:
            val = scores.get(d, 0.0)
            diff_str = ""
            if diff and cat in diff:
                dv = diff[cat].get(d, 0.0)
                diff_str = f" (+{dv})" if dv > 0 else f" ({dv})" if dv < 0 else " (=)"
            parts.append(f"{val}{diff_str}")
        lines.append("  ".join(parts))

    lines.append("")
    overall = aggregate.get("overall", {})
    ov_parts = [f"{'OVERALL':<12}"]
    for d in dims:
        val = overall.get(d, 0.0)
        diff_str = ""
        if diff and "overall" in diff:
            dv = diff["overall"].get(d, 0.0)
            diff_str = f" (+{dv})" if dv > 0 else f" ({dv})" if dv < 0 else " (=)"
        ov_parts.append(f"{val}{diff_str}")
    lines.append("  ".join(ov_parts))

    return "\n".join(lines)


def format_per_question(
    graded_run: GradedRun, category_map: dict[str, str], eval_set_id: str
) -> str:
    eval_set = load_eval_set(eval_set_id)
    case_map = {c.id: c for c in eval_set.cases}

    by_cat: dict[str, list[tuple[GradeResult, str]]] = {}
    for g in graded_run.grades:
        cat = category_map.get(g.case_id, "unknown")
        if cat not in by_cat:
            by_cat[cat] = []
        case = case_map.get(g.case_id)
        question = case.question if case else g.case_id
        by_cat[cat].append((g, question))

    sections: list[str] = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        sections.append(f"\n{cat} ({len(items)} cases)")
        for g, question in items:
            sections.append(f'  "{question}"')
            sections.append(
                f"  Context: {g.context_relevance}/5  "
                f"Groundedness: {g.groundedness}/5  "
                f"Answer: {g.answer_relevance}/5"
            )
            reasons = [
                r for r in [
                    g.context_relevance_reasoning,
                    g.groundedness_reasoning,
                    g.answer_relevance_reasoning,
                ] if r
            ]
            sections.append(f'  Judge: "{"; ".join(reasons)}"')
            sections.append("")

    return "\n".join(sections)


def report_json(
    run_id: str,
    eval_set_id: str,
    test_model: str,
    grade_model: str,
    aggregate: dict[str, dict[str, float]],
    diff: dict[str, dict[str, float]] | None,
    graded_run: GradedRun,
    compared_run_id: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "eval_set_id": eval_set_id,
        "test_model": test_model,
        "grade_model": grade_model,
        "aggregate": aggregate,
    }
    if compared_run_id:
        payload["compared_run_id"] = compared_run_id
        payload["diff"] = diff
    payload["cases"] = [g.model_dump() for g in graded_run.grades]
    return json.dumps(payload, indent=2, ensure_ascii=False)
