"""Query rewriter — first LLM stage in chat retrieval.

Owns the retrieve decision (retrieve true/false), query extraction, mode,
and facet detection (sequence_number / season_number). Runs over an
asymmetric context that excludes all assistant history.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from bibilab.config import AIConfig
from bibilab.pipeline._shared import _call_llm

logger = logging.getLogger(__name__)


class RewriterIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieve: bool
    query: str | None = None
    mode: Literal["narrow", "survey"] | None = None
    sequence_number: int | None = None
    season_number: int | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "RewriterIntent":
        if not self.retrieve:
            if any(v is not None for v in (self.query, self.mode, self.sequence_number, self.season_number)):
                raise ValueError("retrieve=false requires all other fields null")
        else:
            if self.query is None or self.mode is None:
                raise ValueError("retrieve=true requires query and mode")
            if not self.query.strip():
                raise ValueError("retrieve=true requires non-empty query")
        return self


@dataclass(frozen=True)
class PriorUserTurn:
    text: str
    retrieved: bool


_REWRITER_WINDOW = 5

_REWRITER_SYSTEM = """\
You translate a user's chat message into a retrieval intent for a private
video-library notebook. Output JSON only, matching this schema:

{
  "retrieve": boolean,
  "query": string | null,
  "mode": "narrow" | "survey" | null,
  "sequence_number": integer | null,
  "season_number": integer | null
}

retrieve: false when the message is a pure conversational acknowledgment
(e.g. "ok", "嗯", "我懂了", "got it", "thanks") with no new question.
Otherwise true.

When the current message is a continuation signal ("继续", "然后呢",
"再讲讲", "go on", "tell me more") with no new content words:
  - Find the most-recent prior user message tagged retrieve=true.
  - Set query to the EXACT text of that prior message, character for
    character. Do not rewrite, shorten, or summarize it.
  - Copy that prior message's mode and facets verbatim.
  - Do NOT set retrieve=false — the answer model needs fresh excerpts.
  - If no prior user message in the visible window is tagged retrieve=true,
    emit retrieve=false (nothing to continue from).

When retrieve is true:
  query: short search query (1-8 keywords) extracted from the message;
    copy proper nouns and technical terms verbatim.
  mode:
    - "narrow" for single-fact lookups, definitions, specific scenes,
      specific concepts. Default for most questions.
    - "survey" for list-summary or "what happens in episode X" /
      "summarize the whole series" / "compare across sources".
  sequence_number / season_number: parse from THIS MESSAGE only when the
    user explicitly names an episode or season (e.g. "第八集", "episode 3").
    Do not infer from prior turns. Null otherwise.

You see up to 5 prior user messages (each tagged with its retrieve outcome)
as context for pronoun resolution ("上一集", "explain it", "再讲讲那个")
and continuation inheritance ("继续", "然后呢"). You do not see any prior
assistant output.
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_CONTINUATION_RE = re.compile(
    r"^(继续|然后呢|接着|再讲讲|还有呢|还有吗|go on|tell me more|continue|more)$",
    re.IGNORECASE,
)

_ACK_RE = re.compile(
    r"^(ok|okay|k|thanks|thank you|got it|sure|"
    r"嗯|哦|好的?|行|知道了|我懂了|明白|了解了?|了解)$",
    re.IGNORECASE,
)

_SEQ_NUM_RE = re.compile(r"第?(\d+)\s*(集|话|章)", re.IGNORECASE)

_EP_NUM_RE = re.compile(r"(?:episode|ep)\s*(\d+)", re.IGNORECASE)

_SURVEY_RE = re.compile(
    r"(讲了什么|主要讲|概述|总结|全部|所有|"
    r"summarize|summary|overview|what happens|whole|compare)",
    re.IGNORECASE,
)


def build_rewriter_prompt(*, current: str, prior: list[PriorUserTurn]) -> str:
    """Build the COMPLETE rewriter LLM input. Asymmetric-context invariant:
    anything not appended here cannot leak."""
    windowed = prior[-_REWRITER_WINDOW:]
    parts = [_REWRITER_SYSTEM, "\n--- prior user messages ---"]
    for turn in windowed:
        tag = "retrieve=true" if turn.retrieved else "retrieve=false"
        parts.append(f"[{tag}] {turn.text}")
    if not windowed:
        parts.append("(none)")
    parts.append("\n--- current user message ---")
    parts.append(current)
    parts.append("\n--- output JSON only ---")
    return "\n".join(parts)


def parse_rewriter_response(raw: str) -> RewriterIntent | None:
    """Returns None on JSON or invariant failure."""
    text = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.exception("rewriter: invalid JSON (len=%d, tail=%r)", len(raw), raw[-200:])
        return None
    try:
        return RewriterIntent(**data)
    except (ValueError, TypeError):
        logger.exception("rewriter: invariant violation (data=%r)", data)
        return None


def fallback_intent(user_message: str, prior: list[PriorUserTurn] | None = None) -> RewriterIntent:
    """Safe-default intent when the rewriter LLM fails."""
    text = user_message.strip()
    if not text:
        return RewriterIntent(retrieve=False)

    if _ACK_RE.match(text):
        return RewriterIntent(retrieve=False)

    if _CONTINUATION_RE.match(text) and prior:
        for turn in reversed(prior):
            if turn.retrieved:
                return RewriterIntent(retrieve=True, query=turn.text, mode="narrow")
        return RewriterIntent(retrieve=False)

    sequence_number = None
    m = _SEQ_NUM_RE.search(user_message)
    if not m:
        m = _EP_NUM_RE.search(user_message)
    if m:
        sequence_number = int(m.group(1))

    mode = "survey" if _SURVEY_RE.search(user_message) else "narrow"
    return RewriterIntent(retrieve=True, query=user_message, mode=mode, sequence_number=sequence_number)


def run_rewriter(
    *,
    current: str,
    prior: list[PriorUserTurn],
    cfg: AIConfig,
    llm_timeout: int = 30,
    llm_max_tokens: int = 1024,
) -> tuple[RewriterIntent, dict]:
    """Returns (intent, telemetry). telemetry persists at metadata.rag.rewriter."""
    prompt = build_rewriter_prompt(current=current, prior=prior)
    started = time.monotonic()
    fallback = False
    try:
        raw = _call_llm(prompt, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
        intent = parse_rewriter_response(raw)
        if intent is None:
            intent = fallback_intent(current, prior)
            fallback = True
    except Exception:  # noqa: BLE001 - rewriter must always return; fallback owns recovery
        logger.exception("rewriter: LLM call failed; falling back")
        intent = fallback_intent(current, prior)
        fallback = True

    latency_ms = int((time.monotonic() - started) * 1000)
    telemetry = {
        "retrieve": intent.retrieve,
        "mode": intent.mode,
        "sequence_number": intent.sequence_number,
        "season_number": intent.season_number,
        "fallback": fallback,
        "latency_ms": latency_ms,
    }
    return intent, telemetry
