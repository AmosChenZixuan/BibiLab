"""Digest generation - creates summary + keywords from transcript."""

import logging

from pydantic import BaseModel

from bibilab.adapters.base import VideoMeta
from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _STRICT_SUFFIX,
    _TRANSCRIPT_CHAR_LIMIT,
    _call_llm,
    _parse_llm_json_response,
    _resolved_lang,
)

logger = logging.getLogger(__name__)


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
    llm_max_tokens: int = 2048,
) -> DigestResult:
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    if len(transcript_text) > _TRANSCRIPT_CHAR_LIMIT:
        logger.warning("Transcript for %s exceeds ~100K tokens; truncating", meta.video_id)
        transcript_text = transcript_text[:_TRANSCRIPT_CHAR_LIMIT]

    prompt = lang_instruction + "\n\n" + _DIGEST_PROMPT.format(title=meta.title, transcript=transcript_text)

    for attempt in range(3):
        try:
            raw = _call_llm(prompt, cfg, llm_timeout=llm_timeout, llm_max_tokens=llm_max_tokens)
            return _parse_response(raw)
        except Exception as exc:
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
                except Exception as retry_exc:
                    logger.warning(
                        "LLM digest retry failed for %s (attempt %d/3): %s",
                        meta.video_id,
                        attempt + 1,
                        retry_exc,
                    )
                    continue
            else:
                # All 3 retries exhausted — return empty result instead of raising
                logger.warning(
                    "LLM digest exhausted all retries for %s; returning empty result",
                    meta.video_id,
                )
                return DigestResult(summary="", keywords=[])
    # Defensive fallback (should not reach here)
    return DigestResult(summary="", keywords=[])
