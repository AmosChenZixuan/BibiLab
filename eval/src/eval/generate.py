from __future__ import annotations

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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

SOURCE_FACTS_PROMPT = """你是一个研究助理。下面是一段视频的文字稿。请从中提取关键事实。

强约束：
- 只提取原文**逐字出现**或**近义表达**的内容；不要推断、不要补充、不要联想
- 如果文字稿明显空白、纯非语言标记（如 [Music]、[Applause]）、或重复无意义，**所有字段返回空数组**，不要凭领域知识填充
- 中文输出
- 所有字段都有数量上限，超过就**只保留最重要的**

字段定义：
- topics（讨论主题，最多 15 条）：这段内容**整体在讲什么**——抽象的主题/方向/领域。例：「RAG 系统优化」「向量检索原理」。**不是**具体术语（那是 entities）。
- claims（明确断言，最多 20 条，每条 ≤1 句话且贴近原文用词）：原文中**明确说出**的观点、结论、定义、数值、做法。例：「BM25 的 k1 默认值为 1.2」「dense retrieval 在长尾问题上召回率低」。不要把多个 claim 合并成一句长论述。
- entities（专有名词，最多 30 条）：术语、工具名、产品名、人名、模型名、方法名等**有名字的具体事物**。例：「BM25」「HNSW」「OpenAI」「sentence-window chunking」。**不是**主题描述。
- contrasts（同源内部对比，最多 8 条）：**本段视频内部**明确做出的对比——同一文字稿里 A 与 B 被并列比较。例：「文中对比了 BM25 与 dense retrieval 在召回上的差异」。**不要**做跨视频对比（那是下游任务）；**不要**把两个独立 claim 重写成"对比"。
- temporal（时间演变，最多 8 条）：**同一事物**在原文中提到 ≥2 个时间点 / 版本 / 新旧状态的内容。例：「v3.5 → v4 引入 tool use」「2023 推荐 X，2024 改用 Y」「之前用 sliding window，现在改用 semantic chunking」。**单一时间提及**（"今天"、"下次"、单一年份无对比）不算，跳过。

格式要求：
- 只返回**单个 JSON 对象**，不要数组包裹，不要 markdown fences，不要额外文字
- Schema：{{"topics": [...], "claims": [...], "entities": [...], "contrasts": [...], "temporal": [...]}}
- **禁止** {{"sources": [...]}} 这种数组包裹（旧 schema，已废弃）
- 如果某字段没有合格内容，返回空数组 `[]`"""

CATEGORY_PROMPTS: dict[str, str] = {
    "narrow": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**精确查找**类的问题。

精确查找的定义：
- 答案是单个视频某一段落中的具体事实（一个定义、一个数值、一个流程步骤、一个名词解释）
- 不需要汇总多个事实，不需要跨视频对比
- 一句话能问完，一段话能答完

良好示例：
- "什么是 RAG？"
- "向量数据库的索引类型有哪些常用的？"
- "BM25 的 k1 参数默认值是多少？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "X 和 Y 有什么不同？" （属于对比维度）
- "有哪些方法可以做 X？" （属于汇总维度）
- "X 和 Y 有什么关系？" （属于对比维度）

口吻要求：
- 真实用户在聊天框里问的口吻，不要像出题老师
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从提供的要点中**直接抽取或合成**。不要编造要点中没有的事实。如果要点不足以草拟答案，跳过此题。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果确实没有满足条件的题可问，返回 {{"questions": []}}（这是允许的，不要为了凑数编造）。""",

    "broad": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**汇总聚合**类的问题。

汇总聚合的定义：
- 答案需要从多个来源各取一部分，**列举**或**归纳**到一起
- 重点是「全」——把分散在不同视频里的同类信息凑齐
- **不是**对比差异（那是对比维度的事）

良好示例：
- "做向量检索有哪些常用的索引结构？"
- "RAG 系统里常见的 chunking 策略主要有哪几种？"
- "embedding 模型选型时通常会考虑哪些因素？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "BM25 和 dense retrieval 有什么不同？" （属于对比维度——只问差异不要在这里出）
- "什么是 HNSW？" （属于精确查找——单点事实不要在这里出）
- "X 比 Y 好在哪里？" （属于对比维度）

判断标准：好的汇总问题，回答时自然会写「主要有以下几种：A（来源1）、B（来源2）、C（来源3）……」。如果回答只能写「A 和 B 的区别是……」，那就是对比题，放错维度了。

口吻要求：
- 真实用户口吻，问题里可以带聚合提示词："有哪些"、"主要包括"、"常见的几种"、"列出"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从提供的要点中**直接抽取或合成**。不要编造要点中没有的项目。如果可聚合的项目少于 2 个，跳过此题（聚合至少 2 项才有意义）。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果确实没有满足条件的题可问，返回 {{"questions": []}}（这是允许的，不要为了凑数编造）。""",

    "cross_ref": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**跨源综合**类的问题。

跨源综合的定义：
- 回答此题**必须**同时用到 ≥2 个不同来源的信息，缺任何一个都答不完整
- 两种典型形态：
  - **拼接型**：题目涉及多个概念 / 实体 / 维度，每个由不同来源定义或解释。LLM 需要先找全各个来源，再把各自负责的部分拼接成完整答案。
  - **对比型**：≥2 个来源针对同一话题给出不同做法 / 观点 / 参数 / 结论，答案的核心就是「差异」。
- 共同点：单一来源不足以回答；必须跨源。

良好示例：
- 拼接型："BM25 的 k1 默认值在 Lucene 里是多少？" （来源 A 讲 BM25 的 k1，来源 B 讲 Lucene 实现细节）
- 拼接型："用 HNSW 索引存储 OpenAI text-embedding-3-small 的向量需要多少内存？" （来源 A 讲 HNSW 内存模型，来源 B 给出该 embedding 的维度）
- 对比型："dense retrieval 和 BM25 在长尾召回上各有什么短板？" （≥2 个来源分别讨论各自的弱点）

禁止示例（这些属于其它维度，不要在这里生成）：
- "有哪些 chunking 策略？" （属于汇总维度——只列举不需要跨源拼接或对比）
- "什么是 HNSW？" （属于精确查找——单点事实，单来源即可）
- 凭空模板，例如 "请对比 A 和 B"，但要点里只有 A 没有 B 可对比，或 A、B 根本不属于同一话题（属于编造）

判断标准（强制先做）：
1. 在要点里找到此题需要的所有信息位（每个概念 / 每个对比对象）
2. 确认这些信息位**分散在不同来源**，单一来源没有全部覆盖
3. 如果同一来源已经能完整回答（不需要跨源），那是精确查找题或汇总题，**不要**放在此维度
4. 如果要点里找不到任意一个信息位，**跳过此题**，不要为了凑数硬编

口吻要求：
- 真实用户口吻：拼接型问题往往是带定语的具体问法（"X 在 Y 场景下的 Z 是多少"）；对比型常带"和……比"、"……的不同"、"……各自"等词
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从提供的要点中**直接抽取或合成**，对每个信息位标注其来源。不要编造要点中没有的事实。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点中找不到足够的跨源素材，返回 {{"questions": []}}（这是允许的，跨源综合题对素材要求高，没有就是没有）。""",

    "absence": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个**当前来源未涉及**的话题相关问题，用来测试系统在被问到范围外内容时是否会编造。

缺失题的定义：
- 问题问的是一个**合理但不在要点中的话题**——不是无意义乱问，而是「同领域/邻近领域但这批资料没讲」
- 正确答案应该是「这些资料里没有提到 X」（而不是给出虚构事实）

设计要领（强制按顺序做）：
1. 先识别要点的主题领域（例如：RAG / 向量检索 / embedding 模型）
2. 选一个**该领域的子话题或邻近话题**，要点中**没有任何痕迹**（既不在 topics、claims、entities，也不在 contrasts/temporal）
3. 用一个普通用户会问的方式表达；问题本身要看起来"应该可以问"，但要点里查无此事
4. 如果想不出合理的缺失话题，**跳过此题**，不要问完全不沾边的内容（"今天天气如何"不算缺失题，是无关题）

良好示例（假设要点讲 RAG 与 chunking 策略）：
- "你们这里有讲 reranker 模型怎么选吗？" （RAG 领域、相关，但要点未涉及）
- "FAISS 和 Milvus 的性能对比怎么样？" （向量检索领域、相关，但要点未提到具体产品）
- "embedding 模型的微调流程是什么？" （邻近话题，要点没讲微调）

禁止示例：
- "什么是 chunking？" （要点里有 → 这是 narrow 题，不是缺失题）
- "今天天气怎么样？" （完全无关、显然不在范围 → 测的是过滤，不是 RAG 抗幻觉）
- 问要点里某个不存在的具体数值（"BM25 的 k1 默认值是？" 若要点没说）—— 边界模糊，原文可能真的有但被压缩掉了，**避免这类**

口吻要求：
- 真实用户在聊天框里问的口吻，带一点不确定（"你们这里有讲……吗"、"……是不是也涉及"、"……方面有覆盖吗"）
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须形如「当前资料里没有涉及 X。」可以补充 1 句说明为什么这看起来合理但实际未覆盖。**不要**编造虚假事实当作答案。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "当前资料里没有涉及……"}}]}}
如果要点领域窄到想不出合理的缺失话题，返回 {{"questions": []}}（允许，不要硬凑无关题）。""",

    "temporal": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**时效演变**类的问题。

时效演变的定义：
- 题目的核心是**同一事物在不同时间点的变化**（版本演进、推荐方案随时间更替、做法/工具/标准的新旧对比）
- 答案需要识别「旧→新」「过时→当前」「之前 vs 现在」的轴向
- 与跨源对比（cross_ref）的关键区别：cross_ref 是「不同来源同一时刻的差异」，temporal 是「同一事物时间轴上的变化」

设计要领（强制按顺序做）：
1. 先扫描要点中所有 source 的 temporal 字段
2. 找到**同一主题**至少有 2 个时间点的内容（例如同一工具的 v1 与 v3、早期做法 vs 当前做法、过时方案 vs 推荐方案）
3. 如果要点里 temporal 字段稀疏（少于 2 个 source 提到时间）或找不到「同一主题的两个时间点」，**跳过此题**，返回空。不要凭"最新"两字硬编。

良好示例：
- "现在做 RAG 还推荐 BM25 + dense 的混合方案，还是已经被新方法替代了？" （要点提到旧推荐与新做法）
- "OpenAI 的 embedding 模型从 ada-002 到 text-embedding-3 有什么实质改进？" （同模型家族两代）
- "之前流行的 sentence-window chunking 现在还有人用吗，为什么？" （做法时间演变）

禁止示例：
- "BM25 和 dense retrieval 有什么不同？" （属于 cross_ref——同时刻的不同方法对比，不是时间演变）
- "什么是 HNSW？" （属于 narrow——单点查询，没有时间轴）
- "现在最流行的方案是什么？" 但要点里没有任何时间标记 → 凭空编造（禁止）

口吻要求：
- 真实用户口吻，问题里要有时间锚词："现在"、"最新"、"之前"、"已经过时"、"新版"、"从 X 到 Y"、"还推荐用吗"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**直接抽取或合成**时间轴上的两个或多个状态，明示「旧状态：……，新状态：……」或「时间 T1：……，时间 T2：……」。不要编造要点中没有的时间标记或版本信息。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里没有可识别的时间演变素材，返回 {{"questions": []}}（这是允许的，时效演变题对素材要求高，没有就是没有）。""",
}


def _read_transcript(transcript_relpath: str) -> str:
    from bibilab.config import bibilab_home
    p = bibilab_home() / transcript_relpath
    if not p.exists():
        return ""
    return p.read_text()


def _load_per_source(sources: list[dict]) -> list[dict]:
    """Load truncated transcripts per source.

    Per-source word budget = MAX_WORDS / len(sources), min 500. Returns sources
    enriched with a 'transcript' field; missing transcripts are dropped and
    logged to stderr. Splitting per source (vs one giant block) bounds each
    extraction call's input — keeps reasoning-model thinking budget from
    swallowing the whole max_tokens.
    """
    import sys
    word_budget = max(500, MAX_WORDS // max(1, len(sources)))
    loaded: list[dict] = []
    missing: list[str] = []
    for s in sources:
        path = s.get("transcript_path", "")
        raw = _read_transcript(path)
        if not raw:
            missing.append(path or f"<no path for {s.get('id', '?')}>")
            continue
        loaded.append({
            "id": s["id"],
            "title": s.get("title", ""),
            "transcript": _truncate_to_words(raw, word_budget),
        })
    if missing:
        print(f"[generate] missing transcripts ({len(missing)}): {missing}", file=sys.stderr)
    return loaded


_RETRY_HINT = (
    "\n\nYour previous response had invalid JSON. Return ONLY a single JSON "
    'object {"topics": [...], "claims": [...], "entities": [...], '
    '"contrasts": [...], "temporal": [...]} with no fences, no array wrapping, '
    "no extra text."
)


def _persist_failed_raw(kind: str, raw: str) -> Path:
    """Write the raw LLM response to ~/.bibilab/evals/_failed/ for inspection."""
    from bibilab.config import bibilab_home
    from datetime import datetime, timezone

    failed_dir = bibilab_home() / "evals" / "_failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = failed_dir / f"{kind}_{stamp}.txt"
    path.write_text(raw)
    return path


def _try_parse_object(raw: str) -> tuple[dict | None, str | None]:
    stripped = strip_json_fences(raw)
    try:
        data = json.loads(stripped)
        if not isinstance(data, dict):
            return (None, f"Expected JSON object, got {type(data).__name__}; raw prefix: {raw[:200]!r}")
        return (data, None)
    except json.JSONDecodeError as e:
        return (None, f"JSONDecodeError: {e}; raw prefix: {raw[:200]!r}")


def _extract_one_source(source: dict, ai_cfg: Any, language: str = "zh") -> tuple[dict | None, str | None]:
    """Extract facts for a single source. Retries once on malformed JSON.

    Returns (fact_dict | None, error | None). On success, fact_dict carries the
    source id (set by caller, not LLM, to prevent id hallucination). On final
    failure, raw responses persisted to ~/.bibilab/evals/_failed/.
    """
    sid = source["id"]
    body = f"{SOURCE_FACTS_PROMPT}\n\n文字稿内容：\n{source.get('transcript', '')}"
    base_prompt = _with_language(body, language)
    raw = _call_llm(base_prompt, ai_cfg, llm_timeout=120, llm_max_tokens=4096)
    data, err = _try_parse_object(raw)
    if data is None:
        raw2 = _call_llm(base_prompt + _RETRY_HINT, ai_cfg, llm_timeout=120, llm_max_tokens=4096)
        data, err2 = _try_parse_object(raw2)
        if data is None:
            artifact = _persist_failed_raw(
                f"facts_{sid}",
                f"--- attempt 1 ---\n{raw}\n\n--- attempt 2 ---\n{raw2}",
            )
            return (None, f"source {sid}: {err2 or err} (raw persisted: {artifact.name})")

    data["id"] = sid
    return (data, None)


def _extract_facts(
    sources: list[dict],
    ai_cfg: Any,
    language: str = "zh",
    on_progress=None,
) -> tuple[list[dict], list[str]]:
    """Step 1 of 2: extract facts per source in parallel.

    Returns (facts, errors). Partial success allowed — a source whose extraction
    fails twice is skipped and its error appended; the rest still produce facts.
    Caller decides whether `len(facts) == 0` is fatal.
    """
    if not sources:
        return ([], [])

    facts: list[dict] = []
    errors: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {pool.submit(_extract_one_source, s, ai_cfg, language): s for s in sources}
        for fut in as_completed(futures):
            data, err = fut.result()
            if data is not None:
                facts.append(data)
            if err is not None:
                errors.append(err)
            done += 1
            if on_progress is not None:
                on_progress(done, len(sources), len(errors))

    return (facts, errors)


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


def generate_eval_set(
    list_id: str,
    sources: list[dict],
    categories: list[str],
    count: int,
    ai_cfg: Any,
    language: str = "zh",
) -> EvalSet:
    if not categories:
        raise ValueError("No categories specified.")

    selected = random.sample(sources, min(MAX_SOURCES, len(sources)))
    per_source = _load_per_source(selected)
    if not per_source:
        raise ValueError("No transcript content found for this list.")

    word_count = sum(_count_words(s["transcript"]) for s in per_source)
    n_sources = len(per_source)

    task_rows = [("__facts__", "extract facts")] + [(cat, cat) for cat in categories]

    with TaskDashboard(
        "Generate",
        task_rows,
        banner=f"{n_sources} sources, ~{word_count} words, lang={language}",
    ) as dash:
        dash.start("__facts__", status=f"extracting 0/{n_sources}")

        def _on_progress(done: int, total: int, n_errors: int):
            dash.update("__facts__", f"extracting {done}/{total} ({n_errors} failed)")

        facts, facts_errors = _extract_facts(per_source, ai_cfg, language, on_progress=_on_progress)
        facts_block = _format_facts(facts)

        if not facts:
            sample = "; ".join(facts_errors[:3]) if facts_errors else "no transcript content"
            dash.done("__facts__", ok=False, status=f"all {n_sources} failed")
            raise ValueError(f"Fact extraction failed for all sources. Errors: {sample}")

        fact_word_count = _count_words(facts_block)
        compression = word_count // max(1, fact_word_count)
        n_ok = len(facts)
        ok_all = n_ok == n_sources
        status = f"{n_ok}/{n_sources} sources, {fact_word_count} words ({compression}:1)"
        dash.done("__facts__", ok=ok_all, status=status)
        if facts_errors:
            import sys
            print(f"[generate] {len(facts_errors)} source(s) failed extraction:", file=sys.stderr)
            for e in facts_errors:
                print(f"  - {e}", file=sys.stderr)

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
