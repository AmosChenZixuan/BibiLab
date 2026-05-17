"""Digest generation - creates summary + keywords from transcript."""

import json
import logging

import httpx
from pydantic import BaseModel, ValidationError, field_validator

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


class DigestResult(BaseModel):
    summary: str
    keywords: list[str]
    series_name: str | None = None
    sequence_number: int | None = None
    sequence_kind: str | None = None
    season_number: int | None = None

    @field_validator("sequence_number", "season_number", mode="before")
    @classmethod
    def _coerce_int(cls, v: object) -> int | None:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v == int(v):
            return int(v)
        if isinstance(v, str):
            stripped = v.strip()
            try:
                return int(stripped)
            except ValueError:
                pass
        raise ValueError(f"Expected integer, got: {v!r}")

    @field_validator("sequence_kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().lower()
            return cleaned or None
        raise ValueError(f"Expected string or null for sequence_kind, got: {v!r}")


_DIGEST_PROMPT = """\
You are a knowledge extraction assistant. Given a video transcript with timestamps, extract:
1. A single-paragraph abstract of approximately 120–180 words covering the main ideas.
2. Up to 5 short keyword phrases (1–3 words each) reflecting the content density.
3. Series/serial metadata — extract ONLY if explicitly stated in the title or transcript.

   - series_name: the show/program/column identity (e.g. "罗翔说刑法").
     Distinct from the per-video title and the uploader channel name.
     Return null if no series name is apparent.

   - sequence_number: the episode/chapter/part/issue ordinal as an INTEGER.
     Examples: "第8集" → 8, "第12期" → 12, "上篇" → 1, "Chapter 5" → 5.
     Return null if no ordinal is found. Never guess.

   - sequence_kind: short label for what the number represents.
     Typical: "episode", "chapter", "part", "issue", "volume".
     Free-form — use the term that best matches. Return null if unclear.

   - season_number: the season number as an INTEGER if explicitly indicated
     (e.g. "S2" → 2, "第2季" → 2). Usually null.

   If any field is absent or ambiguous, return null. Never guess.

Respond ONLY with valid JSON matching this schema:
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
    if len(result.keywords) > 5:
        result.keywords = result.keywords[:5]
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
        + _DIGEST_PROMPT.format(title=meta.title, transcript=transcript_text)
        + f"\n\n{lang_instruction}\nAll output fields MUST be written in {_LANG_NAME.get(lang, 'English')}."
    )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = _call_llm(prompt, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
            return _parse_response(raw)
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
            logger.warning(
                "LLM digest failed for %s (attempt %d/3): %s",
                meta.video_id,
                attempt + 1,
                exc,
            )
            if attempt < 2:
                try:
                    raw2 = _call_llm(
                        prompt + _STRICT_SUFFIX,
                        cfg,
                        llm_timeout=llm_timeout,
                        llm_max_tokens=llm_max_tokens,
                    )
                    return _parse_response(raw2)
                except (httpx.HTTPError, json.JSONDecodeError, ValidationError) as retry_exc:
                    last_exc = retry_exc
                    logger.warning(
                        "LLM digest retry failed for %s (attempt %d/3): %s",
                        meta.video_id,
                        attempt + 1,
                        retry_exc,
                    )
                    continue

    # All retries exhausted — raise PipelineError instead of silent data loss
    error_msg = f"LLM digest exhausted all retries for {meta.video_id}: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg)
