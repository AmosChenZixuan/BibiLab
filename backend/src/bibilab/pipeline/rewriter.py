"""Query rewriter — first LLM stage in chat retrieval.

Owns the retrieve decision (retrieve true/false), query extraction, mode,
and facet detection (sequence_number / season_number). Runs over an
asymmetric context that excludes all assistant history. See issue #334.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, model_validator

from bibilab.config import AIConfig
from bibilab.pipeline._shared import _call_llm

logger = logging.getLogger(__name__)


class RewriterIntent(BaseModel):
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
  - Copy that message's query, mode, and facets verbatim.
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


def build_rewriter_prompt(*, current: str, prior: list[PriorUserTurn]) -> str:
    """Concatenate system prompt, tagged prior user turns, and the current message.

    Asymmetric-context invariant: this function returns the COMPLETE input
    to the rewriter LLM. Anything not appended here cannot leak.
    """
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
    """Parse rewriter LLM output. Returns None on any failure (invalid JSON,
    Pydantic invariant violation, missing fields)."""
    text = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("rewriter: invalid JSON: %r", raw[:200])
        return None
    try:
        return RewriterIntent(**data)
    except (ValueError, TypeError) as exc:
        logger.warning("rewriter: invariant violation: %s; data=%r", exc, data)
        return None


def fallback_intent(user_message: str) -> RewriterIntent:
    """Construct a safe-default intent when the rewriter LLM fails."""
    return RewriterIntent(retrieve=True, query=user_message, mode="narrow")


def run_rewriter(
    *,
    current: str,
    prior: list[PriorUserTurn],
    cfg: AIConfig,
    llm_timeout: int = 30,
    llm_max_tokens: int = 256,
) -> tuple[RewriterIntent, dict]:
    """Run the rewriter LLM stage.

    Returns (intent, telemetry) where telemetry is the dict persisted at
    metadata.rag.rewriter — includes latency_ms and fallback flag.
    """
    prompt = build_rewriter_prompt(current=current, prior=prior)
    started = time.monotonic()
    fallback = False
    try:
        raw = _call_llm(prompt, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
        intent = parse_rewriter_response(raw)
        if intent is None:
            intent = fallback_intent(current)
            fallback = True
    except Exception:  # noqa: BLE001 - rewriter must always return; downstream owns retry/UX
        logger.exception("rewriter: LLM call failed; falling back")
        intent = fallback_intent(current)
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
