from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bibilab.pipeline._shared import _call_llm

from eval._utils import now_iso, strip_json_fences
from eval.dashboard import TaskDashboard
from eval.models import DEFAULT_FLOOR, EvalCase, EvalSet

MAX_SOURCES = 10
MAX_WORDS_PER_SOURCE = 3000
# Rough CJK char → word budget multiplier. Chinese has no whitespace, so str.split()
# treats a transcript as one giant token. Count chars and divide by ~1.5 to approximate
# token budget against the word-budget ceiling used for whitespace-segmented text.
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

SOURCE_FACTS_PROMPT = """你是一个研究助理。下面是一段视频的文字稿。请用要点概括其内容。

强约束：
- 用 12-20 个要点概括，每个要点一句话，尽量具体（含人物、数值、设定、动作）。
- 只用文字稿中出现的信息，不要补充、推断或联想。
- 时间/版本/新旧信息直接写在要点内部（例：「2023 年推荐 BM25」「v4 新增 tool use」「之前用 A，现在用 B」）。
- 如果文字稿明显空白、纯非语言标记（如 [Music]、[Applause]）或重复无意义，返回空数组。

格式要求：
- 只返回**单个 JSON 对象**，不要数组包裹，不要 markdown fences，不要额外文字
- Schema：{"facts": ["要点1", "要点2", ...]}
- 如果原文没有合格内容，返回 {"facts": []}"""

CATEGORY_PROMPTS: dict[str, str] = {
    "single_fact": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**单点事实**类的问题。

单点事实的定义：
- 答案是某一个来源某一段落里的**一条具体事实**（一个定义、一个数值、一个名词解释、一个具体设定、一句关键台词的含义）
- 一次检索、一个片段即可作答；不需要汇总多条、不需要跨来源、不需要读整段

良好示例：
- "什么是 RAG？"
- "黑塔的守护者叫什么名字？"
- "BM25 的 k1 参数默认值是多少？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "有哪些……？" （属于 enumeration 汇总维度）
- "X 和 Y 有什么不同？" （属于 comparison 对比维度）
- "第三集讲了什么？" （属于 coverage 概览维度）

口吻要求：
- 真实用户在聊天框里问的口吻，不要像出题老师
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从提供的要点中**直接抽取或合成**。不要编造要点中没有的事实。如果要点不足以草拟答案，跳过此题。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果确实没有满足条件的题可问，返回 {{"questions": []}}（这是允许的，不要为了凑数编造）。""",

    "locate": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**定位**类的问题。

定位的定义：
- 用户记得有某件事 / 某句话 / 某个情节，想知道它**出现在哪里**（哪一集、哪一段、大概什么位置）
- 答案的核心是一个**指针**：第几集 / 哪个来源 / 某个时间点，而不是把内容完整复述出来
- 不需要预先知道是哪一集——正是要靠检索找出来

良好示例：
- "人偶一族第一次登场是在哪一集？"
- "讲 chunking 策略的是哪个视频？"
- "槐孟子被提到 365 次那段在哪里？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "第十六集讲了什么？" （属于 coverage——已经指定了集数，问的是内容概览）
- "什么是 RAG？" （属于 single_fact——问事实本身，不是它的位置）

口吻要求：
- 真实用户口吻，带定位词："在哪一集"、"是哪个"、"出现在哪"、"什么时候提到"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**定位到具体来源/集数**。形如「在 X（第 N 集 / 某来源）」。不要编造要点中查无此事的内容。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里没有可定位的素材，返回 {{"questions": []}}（这是允许的，不要硬凑）。""",

    "enumeration": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**枚举**类的问题。

枚举的定义：
- 答案是一份**清单**——把分散在内容各处的同类项**凑齐列全**（人物、规则、步骤、种类、出场角色、提到的工具）
- 重点是「全」：项往往散落在一个来源的多个段落、或多个来源，靠前几个检索片段会漏项，必须把相关段落读全
- **不是**对比差异（那是 comparison），**不是**单条事实（那是 single_fact）

良好示例：
- "人偶一族里出现过哪些角色？"
- "做向量检索有哪些常用的索引结构？"
- "黑塔的规则一共有哪几条？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "什么是 HNSW？" （属于 single_fact——单条事实）
- "BM25 和 dense 有什么不同？" （属于 comparison）

判断标准：好的枚举题，答案自然写成「主要有：A、B、C、D……」。如果只能写「A 和 B 的区别是……」，那是对比题，放错维度了。

口吻要求：
- 真实用户口吻，带枚举词："有哪些"、"一共哪几"、"都包括什么"、"列一下"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**抽取出 ≥3 个同类项**。不要编造要点中没有的项。可枚举项少于 3 个就跳过此题（枚举要成列才有意义）。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里凑不出成列的同类项，返回 {{"questions": []}}（这是允许的，不要硬凑）。""",

    "comparison": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**对比**类的问题。

对比的定义：
- 题目把 **≥2 个对象**（两集的情节、两个角色、两种做法、两个版本）摆在一起问差异 / 异同
- 回答**必须**同时用到每个对象各自的信息——每个对象通常来自不同来源 / 不同集，缺一个就答不完整
- 检索策略是**为每个对象各发一次检索**（每个对象带自己的集数 / 来源限定），再把结果合起来比

良好示例：
- "第三集和第五集的战斗有什么不同？"
- "人偶一族和法师两方的立场分别是什么，区别在哪？"
- "dense retrieval 和 BM25 在长尾召回上各有什么短板？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "有哪些角色？" （属于 enumeration——只列举不比较）
- "什么是 HNSW？" （属于 single_fact）
- 凭空模板「请对比 A 和 B」，但要点里只有 A 没有 B，或 A、B 根本不可比 → 编造，禁止

判断标准（强制先做）：
1. 确认每个对比对象在要点里**都有**实质内容
2. 确认这些内容**分散在不同来源 / 不同集**，需要分别检索
3. 任一对象在要点里查无此事，**跳过此题**

口吻要求：
- 真实用户口吻，带对比词："和……比"、"……的不同"、"……各自"、"哪个更"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**为每个对象各抽取一部分**，标明各自来源 / 集数，再点出差异。不要编造要点中没有的事实。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里找不到两个可比对象，返回 {{"questions": []}}（这是允许的，对比题对素材要求高）。""",

    "multi_hop": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**多跳推理**类的问题。

多跳推理的定义：
- 回答要**分步**：第一步检索得到一个中间结果（某个名字 / 某个值 / 某件事），第二步**用这个中间结果**再检索才能得到最终答案
- 关键特征：不知道第一步的答案就没法发出第二步的检索——两步有**先后依赖**，不能并行
- 与 comparison 的区别：对比是并行（各对象独立检索）；多跳是串行（后一步依赖前一步结果）

良好示例：
- "杀死黑塔守护者的那个人，后来结局如何？" （先查守护者是谁杀的→得到名字→再查那个人的结局）
- "提出混合检索方案的那位作者，他还讲过哪些方法？" （先查方案是谁提的→再查此人其它内容）

禁止示例（这些属于其它维度，不要在这里生成）：
- "A 和 B 有什么不同？" （属于 comparison——并行，不是链式）
- "什么是 RAG？" （属于 single_fact——一步到位）
- 两步之间没有真正依赖、可以并行查的，不要放在此维度

判断标准（强制先做）：
1. 写出第一跳要查什么、得到什么中间结果
2. 确认第二跳**必须**用到第一跳的结果，且这两段信息要点里都有
3. 任一跳在要点里查无此事，**跳过此题**

口吻要求：
- 真实用户口吻，问题里自然带「那个……的人 / 物，他/它……」这种指代前一跳结果的结构
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**走通两跳**——写出中间结果和最终答案，两者都能在要点里找到依据。不要编造。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里找不到可串联的两跳，返回 {{"questions": []}}（这是允许的，多跳题对素材要求高）。""",

    "coverage": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**单集概览**类的问题。

单集概览的定义：
- 用户**指定某一集 / 某一个来源**，问它**整体讲了什么**（剧情梗概、主要内容、这一集发生了什么）
- 答案来自**这一集所有段落的概要综合**，是一份梗概，不是某一条具体事实
- 必须能在要点里定位到**那一集 / 那个来源确实存在**

良好示例：
- "第十六集讲了什么？"
- "介绍一下第一集的故事"
- "讲 chunking 那个视频整体讲了啥？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "人偶一族第一次登场在哪一集？" （属于 locate——问位置，不是某集概览）
- "黑塔守护者叫什么？" （属于 single_fact——单条事实）
- "第三集和第五集有什么不同？" （属于 comparison）

设计要领：
1. 在要点里找一个**标题或内容明确对应某一集 / 某来源**的对象
2. 问它的整体内容，确保要点里该集 / 该来源有足够多的段落可供综合
3. 找不到可对应的单集 / 单来源就**跳过此题**

口吻要求：
- 真实用户口吻："第 N 集讲了什么"、"介绍一下……"、"……主要讲啥"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从该集 / 该来源的要点**综合出一段梗概**。不要编造要点里没有的情节。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里没有可对应某一集 / 某来源的素材，返回 {{"questions": []}}（这是允许的）。""",

    "causal_absent": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个**因果/原因类、但答案在当前内容里缺失**的问题，用来测试系统会不会编造一个原因，还是诚实说明资料里没讲。

因果缺失题的定义：
- 问题问的是「**为什么** X」「X 的原因 / 动机 / 由来是什么」这类因果问题
- X 本身在要点里**有出现**（所以问题看起来理所当然可以问），但**它的原因 / 缘由要点里从未交代**
- 正确答案是「资料里没有解释 X 的原因」，而不是编一个原因出来
- 这是抗幻觉测试：弱模型容易在找不到原因时反复换词重搜，或干脆编一个

设计要领（强制按顺序做）：
1. 在要点里挑一个**确实存在的事件 / 设定 / 结果** X
2. 确认要点**只陈述了 X 发生**，但**没有任何地方解释 X 为什么发生 / 动机 / 由来**
3. 用「为什么 X」「X 是出于什么原因」的方式问出来
4. 如果要点其实交代了原因，那不是缺失题（是 single_fact / multi_hop），**换一个**或跳过

良好示例（假设要点写了「人偶一族与法师结盟」但没写原因）：
- "人偶一族为什么会和法师结盟？"
- "黑塔守护者当初是出于什么原因立下那条规则的？"

禁止示例：
- "为什么用 BM25？" 若要点其实给了理由 → 原因没缺失，不算
- "今天天气为什么这么热？" → 完全不沾边的无关题，不是缺失题
- X 本身要点里就没出现 → 那是普通范围外问题，不是「因果缺失」

口吻要求：
- 真实用户口吻，带因果词："为什么"、"出于什么原因"、"……的动机是"、"怎么会"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须是一句**诚实的弃答**，形如「资料里提到了 X，但没有解释它的原因 / 动机。」**不要**编造一个原因当答案。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "资料里提到了……，但没有解释其原因。"}}]}}
如果要点里每个事件都附带了原因，找不到「有结果无原因」的素材，返回 {{"questions": []}}（这是允许的）。""",

    "temporal": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**时序/顺序**类的问题。

时序的定义：
- 答案的核心是**先后顺序 / 时间线**：几件事按什么顺序发生、某事在另一事之前还是之后、某条发展线随时间怎么演变
- 回答需要把分散的事件按时间 / 集数排好序，而不是只取单点
- 既包含剧情事件的先后（第一集→第三集发生了什么），也包含同一事物随时间的新旧演变（之前用 A，后来改用 B）

良好示例：
- "人偶一族和法师结盟之前，两方之间发生过什么？"
- "黑塔的规则是先立的，还是守护者先出现的？"
- "之前流行的 sentence-window chunking 后来被什么取代了？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "第三集讲了什么？" （属于 coverage——单集概览，没有顺序轴）
- "什么是 HNSW？" （属于 single_fact）
- 问"最新 / 最先"但要点里没有任何时间 / 顺序线索 → 编造，禁止

设计要领（强制先做）：
1. 在要点里找到 ≥2 个**带先后关系 / 时间标记**的事件或状态
2. 确认它们属于同一条线（同一对象的演变，或同一故事里的事件序列）
3. 找不到可排序的素材就**跳过此题**

口吻要求：
- 真实用户口吻，带时序词："先……还是先……"、"之前 / 之后"、"按顺序"、"后来"、"最早"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**排出先后顺序**，明示「先 X，后 Y」或「T1：……，T2：……」。不要编造要点中没有的顺序 / 时间信息。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里没有可识别的时序素材，返回 {{"questions": []}}（这是允许的）。""",

    "entity_profile": """你是一个知识库的普通用户。下面是一些视频内容的要点摘要。你还没有看过原视频。

任务：根据这些要点，提出 {count} 个属于**实体画像**类的问题。

实体画像的定义：
- 围绕**单一实体**（一个角色、一个组织、一个概念、一个工具），问它的**整体情况**——它是谁 / 是什么、做过什么、有哪些特征
- 关键难点：关于该实体的信息**散落在多个段落 / 多个来源**，要把零散提及**聚合**成一份画像
- 与 single_fact 的区别：single_fact 只取一条；entity_profile 要把同一实体的多处提及汇总
- 与 enumeration 的区别：enumeration 列的是同类的**多个项**；entity_profile 聚的是**同一个实体**的多面信息

良好示例：
- "槐孟子是个什么样的角色？"
- "人偶一族这个组织是怎么回事？"
- "BM25 在这些内容里都被用来做什么？"

禁止示例（这些属于其它维度，不要在这里生成）：
- "有哪些角色？" （属于 enumeration——列多个实体，不是聚焦一个）
- "槐孟子在第几集登场？" （属于 locate——问位置）
- "槐孟子的名字怎么写？" （属于 single_fact——单条事实）

设计要领（强制先做）：
1. 在要点里挑一个**被多处提及**的实体（至少 2-3 个不同段落 / 来源都讲到它）
2. 问它的整体画像，确保这些提及能聚合出一份有内容的描述
3. 该实体只被提到一次 → 信息不够画像，**跳过此题**

口吻要求：
- 真实用户口吻："……是谁"、"……是个什么样的"、"……都做了些什么"、"介绍一下……"
- 只用中文
- 不要提到"视频"、"文字稿"、"资料"、"摘要"等字眼

字段约束：
- expected_answer_draft：必须能从要点中**聚合该实体的多处提及**成一段画像。不要编造要点中没有的属性。

只返回 JSON：
{{"questions": [{{"question": "...", "expected_answer_draft": "..."}}]}}
如果要点里没有被多处提及的实体，返回 {{"questions": []}}（这是允许的）。""",
}

# Stratified sampling: every selected category gets DEFAULT_FLOOR questions for
# baseline signal; the failure-prone retrieval shapes (where #523 regressions
# hide) get a surplus on top so they aren't under-sampled at natural frequency.
DEFAULT_WEIGHTS: dict[str, int] = {
    "enumeration": 2,
    "multi_hop": 2,
    "coverage": 2,
    "causal_absent": 2,
}


def resolve_counts(
    categories: list[str],
    floor: int = DEFAULT_FLOOR,
    weights: dict[str, int] | None = None,
) -> dict[str, int]:
    """Per-category question counts: floor for every category, plus a per-type surplus.

    Frequency only modulates the surplus above the floor, so rare-but-failure-prone
    types (enumeration, multi_hop, coverage, causal_absent) still get enough signal.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    return {cat: floor + max(0, weights.get(cat, 0)) for cat in categories}


def _read_transcript(source_id: str) -> str:
    """Load a source's transcript text from bibilab's segments store.

    Returns "" on no segments or a DB-level error (the caller treats both as
    "missing").
    """
    import asyncio
    import sqlite3
    import sys
    from bibilab.pipeline.transcribe import load_transcript_text

    try:
        return asyncio.run(load_transcript_text(source_id, include_time=False))
    except sqlite3.OperationalError:
        # DB-level failure (table missing, DB locked). Programming errors
        # (TypeError, AttributeError) propagate so they're visible.
        print(f"[generate] failed to load transcript for {source_id}", file=sys.stderr)
        return ""


def _load_per_source(sources: list[dict]) -> list[dict]:
    """Load truncated transcripts per source.

    Each source truncated to MAX_WORDS_PER_SOURCE words (flat, not divided).
    Returns sources enriched with a 'transcript' field; missing transcripts are
    dropped and logged to stderr.
    """
    import sys
    loaded: list[dict] = []
    missing: list[str] = []
    for s in sources:
        sid = s.get("id", "")
        raw = _read_transcript(sid)
        if not raw:
            missing.append(sid or "<no id>")
            continue
        loaded.append({
            "id": sid,
            "title": s.get("title", ""),
            "transcript": _truncate_to_words(raw, MAX_WORDS_PER_SOURCE),
        })
    if missing:
        print(f"[generate] missing transcripts ({len(missing)}): {missing}", file=sys.stderr)
    return loaded


_RETRY_HINT = (
    "\n\nYour previous response had invalid JSON. Return ONLY a single JSON "
    'object {"facts": [...]} with no fences, no array wrapping, '
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
    raw = _call_llm(base_prompt, ai_cfg, llm_timeout=120)
    data, err = _try_parse_object(raw)
    if data is None:
        raw2 = _call_llm(base_prompt + _RETRY_HINT, ai_cfg, llm_timeout=120)
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
    lines: list[str] = [
        '（编号仅供内部索引。expected_answer_draft 中禁止引用「第x条」或任何编号，直接写出事实内容即可。）',
        "",
    ]
    for s in facts:
        sid = s.get("id", "?")
        title = s.get("title", "")
        items = s.get("facts", [])
        lines.append(f"=== [{sid}] {title} ===")
        for i, f in enumerate(items, 1):
            lines.append(f"{i}. {f}")
        lines.append("")
    return "\n".join(lines)


def generate_eval_set(
    list_id: str,
    sources: list[dict],
    counts: dict[str, int],
    ai_cfg: Any,
    language: str = "zh",
) -> EvalSet:
    if not counts:
        raise ValueError("No categories specified.")
    categories = list(counts.keys())

    selected = sources[:MAX_SOURCES]
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
            count = counts[category]
            dash.start(category, status=f"generating {count} questions")
            if category not in CATEGORY_PROMPTS:
                dash.done(category, ok=False, status="unknown category")
                return (category, [])
            prompt = CATEGORY_PROMPTS[category].format(count=count)
            full_prompt = _with_language(f"{prompt}\n\n视频内容要点：\n{facts_block}", language)
            raw = _call_llm(full_prompt, ai_cfg, llm_timeout=180)
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
