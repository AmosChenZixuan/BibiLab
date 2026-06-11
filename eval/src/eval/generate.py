from __future__ import annotations

import asyncio as _asyncio
import json
import uuid
from typing import Any

from bibilab.pipeline._shared import _call_llm

from eval._utils import now_iso, strip_json_fences
from eval.claims import Claim, build_claim_pool, load_spans
from eval.dashboard import TaskDashboard
from eval.models import DEFAULT_FLOOR, EvalCase, EvalSet, Evidence
from eval.selection import select

MAX_SOURCES = 10

_LANG_INSTRUCTION = {
    "zh": "",
    "en": "\n\nIMPORTANT: All output (questions, expected_answer_draft, reasoning) MUST be in English. Ignore any Chinese-language instructions above about output language and respond in English only.",
}


def _with_language(prompt: str, language: str) -> str:
    return prompt + _LANG_INSTRUCTION.get(language, "")


# Stratified sampling: every selected category gets DEFAULT_FLOOR questions for
# baseline signal; the failure-prone retrieval shapes (enumeration, multi_hop,
# coverage, causal_absent) get a surplus on top so they aren't under-sampled at
# natural frequency.
DEFAULT_WEIGHTS: dict[str, int] = {
    "enumeration": 2,
    "multi_hop": 2,
    "coverage": 2,
    "causal_absent": 2,
}


def resolve_counts(categories: list[str], floor: int = DEFAULT_FLOOR) -> dict[str, int]:
    """Per-category question counts: floor for every category, plus a per-type surplus.

    Frequency only modulates the surplus above the floor, so rare-but-failure-prone
    types (enumeration, multi_hop, coverage, causal_absent) still get enough signal.
    """
    return {cat: floor + DEFAULT_WEIGHTS.get(cat, 0) for cat in categories}


def _safe_extract_call(prompt: str, ai_cfg: Any) -> tuple[str | None, str | None]:
    """Run one LLM call, turning any failure into an error string.

    A timeout / API error must not crash the surrounding step — the run is
    partial-success by design.
    """
    try:
        return (_call_llm(prompt, ai_cfg), None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


def _spans_for_sources(source_ids: list[str]) -> list[dict]:
    async def _all():
        out = []
        for sid in source_ids:
            out.extend(await load_spans(sid))
        return out
    return _asyncio.run(_all())


def generate_eval_set(
    list_id: str,
    sources: list[dict],
    counts: dict[str, int],
    ai_cfg: Any,
    language: str = "zh",
) -> EvalSet:
    if not counts:
        raise ValueError("No categories specified.")
    selected = sources[:MAX_SOURCES]
    source_ids = [s["id"] for s in selected]

    spans = _spans_for_sources(source_ids)
    if not spans:
        raise ValueError("No section content found for this list.")

    categories = list(counts.keys())
    task_rows = [("__claims__", "extract claims")] + [(c, c) for c in categories]
    cases: list[EvalCase] = []
    with TaskDashboard("Generate", task_rows, banner=f"{len(spans)} spans, lang={language}") as dash:
        dash.start("__claims__", status=f"0/{len(spans)} spans")
        pool, errs = build_claim_pool(spans, ai_cfg, language)
        dash.done("__claims__", ok=bool(pool), status=f"{len(pool)} claims, {len(errs)} span errors")
        if not pool:
            raise ValueError("Claim extraction produced no claims.")

        for cat in categories:
            dash.start(cat, status=f"selecting {counts[cat]}")
            claim_sets = select(cat, pool, counts[cat])
            made = 0
            for cset in claim_sets:
                q, a, err = phrase_question(cat, cset, ai_cfg, language)
                if err or not q:
                    continue
                cases.append(EvalCase(
                    id=str(uuid.uuid4()), category=cat, question=q,
                    expected_answer_draft=a, locked=False, notes="",
                    evidence=[Evidence(source_id=c.source_id, section_seq=c.section_seq,
                                       snippet=c.snippet) for c in cset],
                ))
                made += 1
            dash.done(cat, ok=made > 0, status=f"{made} generated")

    now = now_iso()
    return EvalSet(id=str(uuid.uuid4()), list_id=list_id, created_at=now, updated_at=now, cases=cases)


_PHRASE_SHAPE = {
    "single_fact": "问一个答案就是下面某条事实的单点问题。",
    "locate": "问下面这件事/这句话出现在哪一集 / 哪个来源；答案是它的位置。",
    "coverage": "问'这一集讲了什么'；答案综合下面同一来源的所有事实成一段梗概。",
    "enumeration": "问'有哪些……'；答案把下面事实里的同类项列全。",
    "comparison": "问下面两条（来自不同来源）的差异 / 异同。",
    "multi_hop": "问一个需要先得到下面第一条里的对象、再用它得到第二条答案的两跳问题。",
    "entity_profile": "问下面这个实体整体是怎样的；答案聚合它的多处提及。",
    "temporal": "问下面这些事件的先后顺序 / 时间线。",
    "causal_absent": "问'为什么 X'，X 是下面这条事实里的事件；但下面没有解释原因——答案必须是'资料里提到了 X，但没有解释其原因'，不要编造原因。",
}


def phrase_question(category: str, claims: list[Claim], ai_cfg, language: str = "zh"
                    ) -> tuple[str, str, str | None]:
    facts = "\n".join(f"- {c.text}" for c in claims)
    body = (
        f"你是一个知识库用户，根据下面的事实出一道题。题型要求：{_PHRASE_SHAPE.get(category, '')}\n\n"
        "强约束：用真实用户口吻；只用中文；不要提到'视频/文字稿/资料/摘要'；"
        "expected_answer_draft 必须能从下面事实直接得到（causal_absent 除外，见题型要求）。\n\n"
        f"事实：\n{facts}\n\n"
        '只返回 JSON：{"question":"...","expected_answer_draft":"..."}'
    )
    prompt = _with_language(body, language)
    raw, call_err = _safe_extract_call(prompt, ai_cfg)
    if raw is None:
        return ("", "", call_err)
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as e:
        return ("", "", f"JSONDecodeError: {e}")
    return (str(data.get("question", "")).strip(), str(data.get("expected_answer_draft", "")).strip(), None)
