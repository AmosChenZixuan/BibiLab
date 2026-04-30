"""Digest generation - creates summary + keywords from transcript."""

import json
import logging

import httpx
from pydantic import BaseModel

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

# Sized for thinking-capable models: budget covers reasoning tokens + a ~150-word JSON output.
DIGEST_MAX_TOKENS = 8192


class DigestResult(BaseModel):
    summary: str
    keywords: list[str]


_DIGEST_PROMPT = """\
You are a knowledge extraction assistant. Given a video transcript with timestamps, extract:
1. A single-paragraph abstract of approximately 120–180 words covering the main ideas.
2. Up to 5 short keyword phrases (1–3 words each) reflecting the content density.

Respond ONLY with valid JSON matching this schema:
{{
  "summary": "string",
  "keywords": ["string", ...]
}}

Video title: {title}
Transcript:
{transcript}
"""


def _parse_response(text: str) -> DigestResult:
    result: DigestResult = _parse_llm_json_response(text, DigestResult)
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
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
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
                except (httpx.HTTPError, json.JSONDecodeError) as retry_exc:
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
