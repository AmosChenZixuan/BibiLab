"""Query classification and routing decisions for adaptive RAG."""

import asyncio
import logging
import re

from bibilab.config import BibilabConfig
from bibilab.models._enums import (
    QUERY_TYPE_ANALYTICAL,
    QUERY_TYPE_BREADTH,
    QUERY_TYPE_FACTUAL,
    QueryType,
    RetrievalParams,
)
from bibilab.pipeline._shared import _call_llm

logger = logging.getLogger(__name__)

# The classifier output itself is one short word, but thinking-capable models also consume
# this budget for reasoning tokens. Extended-thinking models need at least 1K for reasoning,
# and the total budget must cover both.
QUERY_CLASSIFICATION_MAX_TOKENS = 8192


CLASSIFICATION_PROMPT = """\
You are a query classifier for a video knowledge base RAG system.
Classify the following user query into exactly one category:

- factual: The query asks for a specific fact, entity, or single-hop answer from
  one or more sources. Examples: "What does video X say about Y?", "When was Z
  mentioned?", "Summarize the main point of video X.", "X是什么？",
  "视频X中为什么提到了Y？"
- breadth: The query asks to survey, list, or count across multiple sources.
  Examples: "Which videos discuss topic X?", "How many sources mention Y?",
  "What are the main themes across the library?", "哪些视频讨论了X？",
  "关于Y有多少个来源？"
- analytical: The query asks to compare, contrast, analyze relationships, or
  reason across multiple concepts. Examples: "How does approach X differ from Y
  across videos?", "Compare the strategies in video A and video B.", "What is
  the relationship between X and Y?", "视频A和B中的方法有什么区别？",
  "X和Y之间的关系是什么？"

Query: {query}
Category:"""


def _build_prompt(query: str) -> str:
    return CLASSIFICATION_PROMPT.format(query=query)


_CLASSIFICATION_RE = re.compile(r"(?<![a-zA-Z])(factual|breadth|analytical)(?![a-zA-Z])", re.IGNORECASE)


def _parse_response(text: str) -> QueryType:
    match = _CLASSIFICATION_RE.search(text)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]
    raise ValueError(f"Unexpected classification response: {text!r}")


async def classify_query(query: str, cfg: BibilabConfig) -> QueryType:
    """Classify a user query into a routing type using an LLM call.

    Falls back to 'factual' on any parse failure or LLM exception so the
    pipeline continues without interruption.
    """
    prompt = _build_prompt(query)
    try:
        raw = await asyncio.to_thread(_call_llm, prompt, cfg.ai, llm_max_tokens=QUERY_CLASSIFICATION_MAX_TOKENS)
        return _parse_response(raw)
    except Exception as exc:
        logger.warning("Query classification failed (%s); falling back to factual", exc)
        return QUERY_TYPE_FACTUAL


_PARAMS_BY_TYPE: dict[QueryType, RetrievalParams] = {
    QUERY_TYPE_FACTUAL: RetrievalParams(depth_per_source=1, top_k=4),
    QUERY_TYPE_ANALYTICAL: RetrievalParams(depth_per_source=4, top_k=12),
    QUERY_TYPE_BREADTH: RetrievalParams(depth_per_source=1, top_k=20),
}


def params_for_type(query_type: QueryType, sources_total: int) -> RetrievalParams:
    """Resolve params for a query type, with list-size-aware adjustments.

    - Breadth on lists with < 3 sources degrades to factual params
      (1 chunk per source on a 1-source list is worse than focused).
    - Breadth top_k is capped at sources_total (asking for 20 chunks from
      a 5-source list is impossible at depth=1 — the diverse selector
      would relax-fallback to duplicates, wasting token budget).
    - Factual uses fixed base params — no scaling with source count.
      Factual queries target specific facts across 1-2 sources; scaling
      admits noise into the LLM context.
    - Analytical top_k is floored at sources_total so all sources can be
      represented, capped at 3× base to bound token usage on very large lists.
    """
    if query_type == QUERY_TYPE_BREADTH and sources_total < 3:
        return _PARAMS_BY_TYPE[QUERY_TYPE_FACTUAL]
    base = _PARAMS_BY_TYPE[query_type]
    if query_type == QUERY_TYPE_BREADTH:
        return RetrievalParams(depth_per_source=base.depth_per_source, top_k=min(base.top_k, sources_total))
    if query_type == QUERY_TYPE_FACTUAL:
        return base
    # Analytical: floored at base.top_k so all sources can be represented,
    # capped at min(sources_total, base.top_k * 3) to bound token usage.
    top_k = max(base.top_k, min(sources_total, base.top_k * 3))
    return RetrievalParams(depth_per_source=base.depth_per_source, top_k=top_k)
