"""Query classification and routing decisions for adaptive RAG."""

import asyncio
import logging

from bibilab.config import BibilabConfig
from bibilab.models._enums import (
    QUERY_TYPE_ANALYTICAL,
    QUERY_TYPE_BREADTH,
    QUERY_TYPE_FACTUAL,
    QueryType,
)
from bibilab.pipeline._shared import _call_llm

logger = logging.getLogger(__name__)

# The classifier output itself is one short word, but thinking-capable models also consume
# this budget for reasoning tokens. Keep generous headroom or the response gets truncated to empty.
QUERY_CLASSIFICATION_MAX_TOKENS = 4096


CLASSIFICATION_PROMPT = """\
You are a query classifier for a video knowledge base RAG system.
Classify the following user query into exactly one category:

- factual: The query asks for a specific fact, entity, or single-hop answer from
  one or more sources. Examples: "What does video X say about Y?", "When was Z
  mentioned?", "Summarize the main point of video X."
- breadth: The query asks to survey, list, or count across multiple sources.
  Examples: "Which videos discuss topic X?", "How many sources mention Y?",
  "What are the main themes across the library?"
- analytical: The query asks to compare, contrast, analyze relationships, or
  reason across multiple concepts. Examples: "How does approach X differ from Y
  across videos?", "Compare the strategies in video A and video B.", "What is
  the relationship between X and Y?"

Query: {query}
Category:"""


def _build_prompt(query: str) -> str:
    return CLASSIFICATION_PROMPT.format(query=query)


def _parse_response(text: str) -> QueryType:
    cleaned = text.strip().strip('"').lower()
    if cleaned in (QUERY_TYPE_FACTUAL, QUERY_TYPE_BREADTH, QUERY_TYPE_ANALYTICAL):
        return cleaned  # type: ignore[return-value]
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
