"""Digest generation - creates summary + keywords from transcript."""

import json
import logging
import math

import httpx
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator, model_validator

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _LANG_NAME,
    _STRICT_SUFFIX,
    _call_llm,
    _parse_llm_json_response,
    _resolved_lang,
)
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

# Sized for thinking-capable models: budget covers reasoning tokens + a ~150-word
# JSON output. Reasoning models can consume 12K+ tokens thinking about a long
# transcript, so we need ample headroom. The JSON output itself is bounded by the
# schema and can't go haywire.
DIGEST_MAX_TOKENS = 32768

# Keyword count: feeds the digest chip UI and the chat source-scoping prompt
# (routers/chat.py builds "[N] Title (keywords)"). Kept short and topical so
# both consumers stay cheap; raised from 5 to widen query-topic coverage.
_MAX_KEYWORDS = 8


class DigestResult(BaseModel):
    summary: str
    keywords: list[str]
    series_name: str | None = None
    sequence_number: int | None = None
    sequence_kind: str | None = None
    season_number: int | None = None

    @field_validator("sequence_number", "season_number", mode="before")
    @classmethod
    def _coerce_int(cls, v: object, info: ValidationInfo) -> int | None:
        # Facets are best-effort: an unparseable or out-of-range value degrades
        # to None (logged) rather than raising. A null facet is "unknown" and
        # falls back to normal behavior; a bad facet must never abort the digest.
        if v is None:
            return None
        if isinstance(v, bool):  # bool is an int subclass; JSON true/false is not an ordinal
            n = None
        elif isinstance(v, int):
            n = v
        elif isinstance(v, float) and math.isfinite(v) and v == int(v):
            n = int(v)
        elif isinstance(v, str):
            try:
                n = int(v.strip())
            except ValueError:
                n = None
        else:
            n = None
        if n is None or n < 1:
            logger.warning("digest: dropping unusable %s=%r", info.field_name, v)
            return None
        return n

    @field_validator("series_name", "sequence_kind", mode="before")
    @classmethod
    def _clean_str_facet(cls, v: object, info: ValidationInfo) -> str | None:
        # Best-effort like the int facets: a non-string or blank value degrades
        # to None (logged), never raises — a bad facet must not abort the
        # digest. sequence_kind is lowercased (a label); series_name is not (a
        # proper noun).
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()
            if info.field_name == "sequence_kind":
                cleaned = cleaned.lower()
            return cleaned or None
        logger.warning("digest: dropping unusable %s=%r", info.field_name, v)
        return None

    @model_validator(mode="after")
    def _require_kind_with_number(self) -> "DigestResult":
        # sequence_number and sequence_kind are a unit: a number with no label
        # is unrenderable ("8 of what?"), a label with no number is meaningless.
        # If the LLM emits exactly one, drop both rather than persist a half-facet.
        if (self.sequence_number is None) != (self.sequence_kind is None):
            logger.warning(
                "digest: dropping mismatched sequence facet (number=%r, kind=%r)",
                self.sequence_number,
                self.sequence_kind,
            )
            self.sequence_number = None
            self.sequence_kind = None
        return self


_DIGEST_PROMPT = """\
Extract structured metadata from the video below. Return ONE JSON object and
nothing else. Each field is described once; follow its rule exactly.

summary (string)
  One paragraph, ~120-180 words, covering the main ideas.

keywords (list of strings, up to {max_keywords}, each 1-4 words)
  The main subjects the video covers, taken from title and transcript. Each
  keyword must be a self-contained topic a viewer could ask a question about
  on its own: a concrete dish, product, person, place, work, concept,
  theory, or technique.
  Exclude three kinds of non-topics:
    - format / meta words   e.g. 教程, 讲解, 合集, vlog, 视频
    - standalone modifiers  e.g. 麻辣, 韩式, 高清
    - labels too generic to identify this video   e.g. 美食, 知识
  Avoid near-duplicates (keep the most specific form only).
  Examples across genres (titles shown for brevity; the transcript adds more):
    "10分钟搞定！韩式炸鸡 + 辣鸡面"   -> 韩式炸鸡, 辣鸡面   (drop "10分钟搞定")
    "罗翔讲刑法：正当防卫的边界"      -> 罗翔, 刑法, 正当防卫   (drop "讲")
    "iPhone 16 Pro 深度评测 影像与续航" -> iPhone 16 Pro, 影像, 续航 (drop "深度评测")

series_name (string or null)
  The show / column identity, distinct from this video's title and the
  uploader name. e.g. "罗翔说刑法". null if none is stated.

sequence_number (integer or null)
  The video's ordinal within the series.
  e.g. "第8集"->8, "第12期"->12, "Chapter 5"->5, "上篇/中篇/下篇"->1/2/3.
  null if no ordinal is stated.

sequence_kind (string or null)
  What sequence_number counts, free-form: episode, chapter, part, issue,
  volume, ... null if unclear.

season_number (integer or null)
  e.g. "S2"->2, "第2季"->2. Usually null.

Rule for the four series_* fields: extract ONLY a value explicitly stated in
the title or transcript. Never guess. When unsure, use null - a wrong value
silently hides correct content later, while null just falls back to normal.

Output JSON in exactly this shape:
{{
  "summary": "string",
  "keywords": ["string", ...],
  "series_name": "string | null",
  "sequence_number": "integer | null",
  "sequence_kind": "string | null",
  "season_number": "integer | null"
}}

Video title: {title}
Transcript:
{transcript}
"""


def _parse_response(text: str) -> DigestResult:
    result: DigestResult = _parse_llm_json_response(text, DigestResult)
    result.keywords = result.keywords[:_MAX_KEYWORDS]
    return result


def digest(
    transcript_text: str,
    meta: VideoMeta,
    cfg: AIConfig,
    output_language: str = "ui",
    ui_lang: str | None = None,
    llm_timeout: int = 120,
    llm_max_tokens: int = DIGEST_MAX_TOKENS,
) -> DigestResult:
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    char_limit = cfg.transcript_char_limit
    if len(transcript_text) > char_limit:
        logger.warning("Transcript for %s exceeds %d chars; truncating", meta.video_id, char_limit)
        transcript_text = transcript_text[:char_limit]

    prompt = (
        lang_instruction
        + "\n\n"
        + _DIGEST_PROMPT.format(title=meta.title, transcript=transcript_text, max_keywords=_MAX_KEYWORDS)
        + f"\n\n{lang_instruction}\nAll output fields MUST be written in {_LANG_NAME.get(lang, 'English')}."
    )

    # One plain attempt, then strict-suffix retries. Linear so the attempt
    # count is unambiguous. Only summary/keywords failures land here now —
    # facets degrade to None in the validators and never raise.
    prompts = [prompt, prompt + _STRICT_SUFFIX, prompt + _STRICT_SUFFIX]
    last_exc: Exception | None = None
    for attempt, p in enumerate(prompts, start=1):
        try:
            raw = _call_llm(p, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
            return _parse_response(raw)
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
            logger.warning(
                "LLM digest failed for %s (attempt %d/%d): %s",
                meta.video_id,
                attempt,
                len(prompts),
                exc,
            )

    # All retries exhausted — raise PipelineError instead of silent data loss
    error_msg = f"LLM digest exhausted all retries for {meta.video_id}: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg)
