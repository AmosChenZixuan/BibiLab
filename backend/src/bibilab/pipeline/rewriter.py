"""Query rewriter — first LLM stage in chat retrieval.

Failure policy: retry until schema-valid intent or budget exhausted. No regex
fallback — a degraded answer is worse than a surfaced error.
"""

import json
import logging
import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bibilab.config import AIConfig
from bibilab.models._enums import Mode
from bibilab.pipeline._shared import _call_llm, _parse_llm_json_response
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)


class RewriterIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieve: bool
    query: str | None = None
    mode: Mode | None = None
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


class RewriterError(PipelineError):
    """Rewriter exhausted retry budget without a schema-valid response."""


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
  - Treat that prior message as the question to retrieve for. Apply the
    same query-construction rules below (narrow vs survey) to its text.
  - Copy that prior message's mode and facets verbatim.
  - Do NOT set retrieve=false — the answer model needs fresh excerpts.
  - If no prior user message in the visible window is tagged retrieve=true,
    emit retrieve=false (nothing to continue from).

When retrieve is true:
  query: a search query optimized for hybrid BM25 + vector retrieval over
    spoken-language transcripts. Construction rules depend on mode:

    - narrow mode: include EVERY content noun, number, and proper noun
      from the message. Drop only stopwords (的, 了, 是, 吗, what, the, is,
      a, an) and pure interrogatives (多少, 哪, how, which) when they carry
      no lexical weight. Do not collapse the question down to its "main"
      noun — every content word in the question is potentially the one
      that anchors the answer passage, so all must survive into the query.
      "拉格朗日点稳定性是怎么证明的" → "拉格朗日点 稳定性 证明".

    - survey mode: the goal is to widen lexical coverage of the TOPIC, not
      to guess which instances the source covers. NEVER invent specific
      named entities the user did not mention — you have not read the
      source and your guesses will miss. Instead:
        a) keep the topic anchor noun from the question (面食, 哲学, 战争);
        b) add 3-6 synonyms, near-synonyms, or hypernyms of the topic
           anchor and of the category word (流派 → 学派, 思想, 主义;
           做法 → 做, 制作, 烹饪, 食谱);
        c) drop pure enumeration words ("哪些", "有什么", "list", "kinds
           of") — they carry no lexical weight.
      "有哪些面食做法" → "面食 面条 面 主食 面粉 做 制作 烹饪".
      "讲了哪些哲学流派" → "哲学 流派 学派 思想 主义 哲学家".

    Keep proper nouns and technical terms verbatim in both modes.

  mode:
    - "narrow" for single-fact lookups, definitions, specific scenes,
      specific concepts. Default for most questions.
    - "survey" for list-summary or "what happens in episode X" /
      "summarize the whole series" / "compare across sources" /
      "有哪些 X" / "列举一下 Y".
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


def build_rewriter_prompt(*, current: str, prior: list[PriorUserTurn]) -> str:
    """Asymmetric-context invariant: this returns the COMPLETE rewriter input.
    Anything not appended here cannot leak from prior turns."""
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
    """None on JSON or invariant failure."""
    try:
        return _parse_llm_json_response(raw, RewriterIntent)
    except json.JSONDecodeError:
        logger.warning("rewriter: invalid JSON (len=%d, tail=%r)", len(raw), raw[-200:])
        return None
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("rewriter: invariant violation (%s)", exc)
        return None


def run_rewriter(
    *,
    current: str,
    prior: list[PriorUserTurn],
    cfg: AIConfig,
    llm_timeout: int = 60,
    # Output JSON is ~100 tokens, but thinking-capable models can burn the
    # entire budget on reasoning before emitting text (stop_reason=max_tokens
    # with empty content). Generous budget covers thinking + JSON.
    llm_max_tokens: int = 4096,
) -> tuple[RewriterIntent, dict]:
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
            logger.info(
                "rewriter intent: retrieve=%s query=%r mode=%s seq=%s season=%s attempts=%d latency_ms=%d",
                intent.retrieve,
                intent.query,
                intent.mode,
                intent.sequence_number,
                intent.season_number,
                attempt + 1,
                latency_ms,
            )
            return intent, telemetry
        last_error = ValueError("rewriter returned schema-invalid JSON")

    logger.error("rewriter: exhausted %d attempts; last error: %s", _REWRITER_MAX_ATTEMPTS, last_error)
    raise RewriterError("rewriter exhausted retry budget") from last_error
