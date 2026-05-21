"""Query rewriter — first LLM stage in chat retrieval.

Owns the retrieve decision (retrieve true/false), query extraction, mode,
and facet detection (sequence_number / season_number). Runs over an
asymmetric context that excludes all assistant history.

Failure policy: retry until the LLM returns a schema-valid intent or the
retry budget is exhausted. No regex fallback — a degraded answer is worse
than a surfaced error.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

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


class RewriterError(RuntimeError):
    """Rewriter exhausted retry budget without a schema-valid response.

    Raised by run_rewriter on terminal failure. The outer chat handler
    surfaces this as an SSE error event — no degraded fallback.
    """


_REWRITER_WINDOW = 5
_REWRITER_MAX_ATTEMPTS = 3
_REWRITER_BACKOFF_SECONDS = (0.0, 1.0, 3.0)

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

_CORRECTION_SUFFIX = (
    "\n\nYour previous output was not valid JSON or violated the schema. "
    "Return ONLY a JSON object matching the schema above. No prose, no fences."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


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
        logger.warning("rewriter: invalid JSON (len=%d, tail=%r)", len(raw), raw[-200:])
        return None
    try:
        return RewriterIntent(**data)
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("rewriter: invariant violation (%s, data=%r)", exc, data)
        return None


def run_rewriter(
    *,
    current: str,
    prior: list[PriorUserTurn],
    cfg: AIConfig,
    llm_timeout: int = 30,
    llm_max_tokens: int = 1024,
) -> tuple[RewriterIntent, dict]:
    """Retry until the LLM returns a schema-valid intent or budget is exhausted.

    Provider errors (timeouts, 429, 5xx) are retried with exponential backoff.
    JSON / schema errors are retried with a correction suffix appended to the
    prompt. On exhaustion, raises RewriterError — the caller MUST surface this
    to the user; degraded answers are not allowed.
    """
    base_prompt = build_rewriter_prompt(current=current, prior=prior)
    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(_REWRITER_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(_REWRITER_BACKOFF_SECONDS[attempt])
        prompt = base_prompt if attempt == 0 else base_prompt + _CORRECTION_SUFFIX
        try:
            raw = _call_llm(prompt, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
        except Exception as exc:  # noqa: BLE001 - retry every provider error class
            last_error = exc
            logger.warning("rewriter: LLM call failed (attempt %d/%d): %s", attempt + 1, _REWRITER_MAX_ATTEMPTS, exc)
            continue

        intent = parse_rewriter_response(raw)
        if intent is not None:
            latency_ms = int((time.monotonic() - started) * 1000)
            telemetry = {
                "retrieve": intent.retrieve,
                "mode": intent.mode,
                "sequence_number": intent.sequence_number,
                "season_number": intent.season_number,
                "attempts": attempt + 1,
                "latency_ms": latency_ms,
            }
            return intent, telemetry
        last_error = ValueError("rewriter returned schema-invalid JSON")

    logger.error("rewriter: exhausted %d attempts; last error: %s", _REWRITER_MAX_ATTEMPTS, last_error)
    raise RewriterError("rewriter exhausted retry budget") from last_error
