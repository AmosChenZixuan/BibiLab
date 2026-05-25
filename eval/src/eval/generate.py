from __future__ import annotations

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bibilab.pipeline._shared import _call_llm

from eval._utils import now_iso, strip_json_fences
from eval.dashboard import TaskDashboard
from eval.models import EvalCase, EvalSet

MAX_SOURCES = 10
MAX_WORDS = 20000
# Rough CJK char → word budget multiplier. Chinese has no whitespace, so str.split()
# treats a transcript as one giant token. Count chars and divide by ~1.5 to approximate
# token budget against the same MAX_WORDS ceiling used for whitespace-segmented text.
_CJK_CHARS_PER_WORD = 1.5


def _count_words(text: str) -> int:
    """Word count that handles CJK text (no whitespace) via char-based fallback."""
    whitespace_tokens = len(text.split())
    if whitespace_tokens >= max(1, len(text) // 10):
        return whitespace_tokens
    return int(len(text) / _CJK_CHARS_PER_WORD)


def _truncate_to_words(text: str, word_budget: int) -> str:
    words = text.split()
    if len(words) >= max(1, len(text) // 10):
        if len(words) > word_budget:
            return " ".join(words[:word_budget]) + "\n[...truncated...]"
        return text
    char_budget = int(word_budget * _CJK_CHARS_PER_WORD)
    if len(text) > char_budget:
        return text[:char_budget] + "\n[...truncated...]"
    return text

_LANG_INSTRUCTION = {
    "zh": "",
    "en": "\n\nIMPORTANT: All output (questions, expected_answer_draft, reasoning) MUST be in English. Ignore any Chinese-language instructions above about output language and respond in English only.",
}


def _with_language(prompt: str, language: str) -> str:
    return prompt + _LANG_INSTRUCTION.get(language, "")

FACTS_PROMPT = """你是一个研究助理。下面是几段视频的文字稿。请从每段文字稿中提取关键事实。

严格要求：
- 只提取文字稿中明确陈述的内容，不要推断、不要补充、不要联想
- 每条事实必须能在原文中找到对应
- 中文输出

对每个来源提取：
- topics: 讨论了哪些主题/话题
- claims: 明确提出的观点、结论、定义（"X是Y"、"X导致Y"这类断言）
- entities: 提到的专有名词、术语、人名、工具名、方法名
- contrasts: 文中提到的对比、不同观点或方案差异
- temporal: 涉及时间、版本、时效性的信息

只返回 JSON，不要任何额外文字：
{{"sources": [{{"id": "...", "topics": ["..."], "claims": ["..."], "entities": ["..."], "contrasts": ["..."], "temporal": ["..."]}}]}}
如果某来源没有某类信息，对应字段返回空列表。"""

CATEGORY_PROMPTS: dict[str, str] = {
    "narrow": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个你真正会问的具体问题。

要求：
- 问题类型：精确查找（什么是X / X是什么 / X和Y有什么关系）
- 答案应该在某个视频的某一段落中就能找到，不要问需要跨视频综合的问题
- 用真实用户的口吻提问，不要像出题老师
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼
- 每个问题要像聊天时随口问的

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": ["id"]}}]}}
如果确实问不出好问题，返回：{{"questions": []}}""",

    "broad": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个你真正会问的综合性问题。

要求：
- 问题类型：汇总归纳（有哪些方法 / 什么样的流程 / 涉及哪些方面）
- 需要综合多个视频的内容才能完整回答
- 问题里要有自然的筛选条件（"列出至少三个"、"主要有哪些"、"在哪些情况下"）
- 用真实用户的口吻提问，不要像出题老师
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": ["id1","id2"]}}]}}
如果确实问不出好问题，返回：{{"questions": []}}""",

    "cross_ref": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，你想对比不同视频中讲的内容。提出 {count} 个需要跨视频对比的问题。

要求：
- 问题类型：对比分析（A和B的做法有什么不同 / X和Y分别怎么看某个问题）
- 确实需要对比两个不同视频中的观点或信息才能回答
- 问题本身就是对比式的，不是在单方面询问
- 用真实用户的口吻提问
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": ["id1","id2"]}}]}}
如果确实没有可对比的内容，返回：{{"questions": []}}""",

    "ambiguous": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：你正在打字问问题，但你没说清楚。根据这些要点，提出 {count} 个模糊或不完整的问题。

要求：
- 问题特征：用词模糊（"那个东西"、"之前说的那个"）、缺少上下文、可以有两种以上理解
- 从这些要点中，能够推断出用户真正想问什么（只是用户没表达清楚）
- 用真实用户的口吻，就像在聊天框里随手打的
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": ["id"]}}]}}
如果问不出模糊问题，返回：{{"questions": []}}""",

    "absence": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：你隐约记得这些视频可能讲过某件事，但不确定。根据这些要点，提出 {count} 个关于"是否提到了X"的问题。

要求：
- 一半的问题：要点中明确涉及的话题（答案应该是"提到了"）
- 一半的问题：和要点中的话题相关、但要点中没有明确提到的（答案应该是"没有提到"）
- 用真实用户的口吻，就像在确认记忆
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": []}}]}}""",

    "temporal": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：这些视频可能有一些时效性内容。根据这些要点，提出 {count} 个关于"最新/当前/现在"的问题。

要求：
- 问题类型：时效性（现在还用这个方法吗 / 最新版本的实现是什么 / 当前推荐哪个方案）
- 要点中确实有不同版本或不同时间点的内容可以对比
- 用真实用户的口吻提问
- 只用中文，不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼
- 如果要点中没有明显的时效性信息，返回空

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "...", "expected_sources": ["id"]}}]}}
如果确实没有时效性内容，返回：{{"questions": []}}""",
}


def _read_transcript(transcript_relpath: str) -> str:
    from bibilab.config import bibilab_home
    p = bibilab_home() / transcript_relpath
    if not p.exists():
        return ""
    return p.read_text()


def _load_sources(sources: list[dict]) -> str:
    """Per-source word budget = MAX_WORDS / len(sources), min 500."""
    import sys
    word_budget = max(500, MAX_WORDS // max(1, len(sources)))
    lines: list[str] = []
    missing: list[str] = []
    for s in sources:
        path = s.get("transcript_path", "")
        raw = _read_transcript(path)
        if not raw:
            missing.append(path or f"<no path for {s.get('id', '?')}>")
            continue
        raw = _truncate_to_words(raw, word_budget)
        lines.append(f"=== [{s['id']}] {s.get('title', '')} ===")
        lines.append(raw)
        lines.append("")
    if missing:
        print(f"[generate] missing transcripts ({len(missing)}): {missing}", file=sys.stderr)
    return "\n".join(lines)


def _extract_facts(source_block: str, ai_cfg: Any, language: str = "zh") -> tuple[list[dict], str | None]:
    """Extract structured facts from transcripts (step 1 of 2).

    Returns (sources, parse_error). parse_error is None on success; otherwise a
    short diagnostic including the LLM's raw output prefix so a downstream caller
    can surface *why* extraction failed rather than just "empty result".
    """
    full_prompt = _with_language(f"{FACTS_PROMPT}\n\n文字稿内容：\n{source_block}", language)
    raw = _call_llm(full_prompt, ai_cfg, llm_timeout=180, llm_max_tokens=16384)
    stripped = strip_json_fences(raw)
    try:
        data = json.loads(stripped)
        return (data.get("sources", []), None)
    except json.JSONDecodeError as e:
        return ([], f"JSONDecodeError: {e}; raw prefix: {raw[:200]!r}")


def _format_facts(facts: list[dict]) -> str:
    """Format extracted facts into a compact readable block for question generation."""
    lines: list[str] = []
    for s in facts:
        sid = s.get("id", "?")
        lines.append(f"=== [{sid}] ===")
        for field, label in [
            ("topics", "主题"),
            ("claims", "观点"),
            ("entities", "术语"),
            ("contrasts", "对比"),
            ("temporal", "时效"),
        ]:
            items = s.get(field, [])
            if items:
                lines.append(f"{label}: {'; '.join(items)}")
        lines.append("")
    return "\n".join(lines)


async def generate_eval_set(
    list_id: str,
    categories: list[str],
    count: int,
    ai_cfg: Any,
    language: str = "zh",
) -> EvalSet:
    from bibilab.db import get_sources_for_list

    if not categories:
        raise ValueError("No categories specified.")

    rows = await get_sources_for_list(list_id)
    all_sources = [dict(r) for r in rows]
    selected = random.sample(all_sources, min(MAX_SOURCES, len(all_sources)))
    source_block = _load_sources(selected)
    if not source_block:
        raise ValueError("No transcript content found for this list.")

    word_count = _count_words(source_block)

    task_rows = [("__facts__", "extract facts")] + [(cat, cat) for cat in categories]

    with TaskDashboard(
        "Generate",
        task_rows,
        banner=f"{len(selected)} sources, ~{word_count} words, lang={language}",
    ) as dash:
        dash.start("__facts__", status="calling LLM")
        facts, facts_err = _extract_facts(source_block, ai_cfg, language)
        facts_block = _format_facts(facts)
        if not facts_block.strip():
            status = facts_err or "empty result"
            dash.done("__facts__", ok=False, status=status[:80])
            if facts_err:
                raise ValueError(f"Fact extraction failed to parse LLM response. {facts_err}")
            raise ValueError("Fact extraction produced empty results. Check transcript quality or LLM config.")
        fact_word_count = _count_words(facts_block)
        dash.done(
            "__facts__",
            ok=True,
            status=f"{fact_word_count} words ({word_count // max(1, fact_word_count)}:1 compression)",
        )

        def _generate_one(category: str) -> tuple[str, list[dict]]:
            dash.start(category, status=f"generating {count} questions")
            if category not in CATEGORY_PROMPTS:
                dash.done(category, ok=False, status="unknown category")
                return (category, [])
            prompt = CATEGORY_PROMPTS[category].format(count=count)
            full_prompt = _with_language(f"{prompt}\n\n视频内容要点：\n{facts_block}", language)
            raw = _call_llm(full_prompt, ai_cfg, llm_timeout=180, llm_max_tokens=16384)
            stripped = strip_json_fences(raw)
            try:
                data = json.loads(stripped)
                qs = data.get("questions", [])
                dash.done(category, ok=bool(qs), status=f"{len(qs)} generated")
                return (category, qs)
            except json.JSONDecodeError as e:
                dash.done(category, ok=False, status=f"JSON parse: {e}; raw: {raw[:60]!r}"[:80])
                return (category, [])

        cases: list[EvalCase] = []
        with ThreadPoolExecutor(max_workers=len(categories)) as pool:
            futures = {pool.submit(_generate_one, cat): cat for cat in categories}
            for future in as_completed(futures):
                cat, qs = future.result()
                for q in qs:
                    cases.append(
                        EvalCase(
                            id=str(uuid.uuid4()),
                            category=cat,
                            question=q.get("question", ""),
                            expected_answer_draft=q.get("expected_answer_draft", ""),
                            expected_sources=q.get("expected_sources", []),
                            locked=False,
                            notes="",
                        )
                    )

    now = now_iso()
    return EvalSet(
        id=str(uuid.uuid4()),
        list_id=list_id,
        created_at=now,
        updated_at=now,
        cases=cases,
    )
