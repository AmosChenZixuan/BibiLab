"""LLM knowledge extraction step."""

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
from bibilab.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)


class KeyPoint(BaseModel):
    timestamp: str
    text: str


class ExtractionResult(BaseModel):
    title: str
    summary: str
    key_points: list[KeyPoint]


_EXTRACTION_PROMPT = """\
You are a knowledge extraction assistant. Given a video transcript with timestamps, extract:
1. A concise title (if different from the provided title, otherwise reuse it).
2. A 3-5 sentence summary covering the main ideas.
3. 5-15 key points, each with a timestamp in [HH:MM:SS] format and a clear sentence.

Respond ONLY with valid JSON matching this schema:
{{
  "title": "string",
  "summary": "string",
  "key_points": [{{"timestamp": "[HH:MM:SS]", "text": "string"}}]
}}

Video title: {title}
Transcript:
{transcript}
"""


def _parse_response(text: str) -> ExtractionResult:
    return _parse_llm_json_response(text, ExtractionResult)


def extract_knowledge(
    transcript_text: str,
    meta: VideoMeta,
    cfg: AIConfig,
    output_language: str = "ui",
    ui_lang: str | None = None,
) -> ExtractionResult:
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    if len(transcript_text) > _TRANSCRIPT_CHAR_LIMIT:
        logger.warning("Transcript for %s exceeds ~100K tokens; truncating", meta.video_id)
        transcript_text = transcript_text[:_TRANSCRIPT_CHAR_LIMIT]

    prompt = lang_instruction + "\n\n" + _EXTRACTION_PROMPT.format(title=meta.title, transcript=transcript_text)

    raw = _call_llm(prompt, cfg)
    try:
        return _parse_response(raw)
    except Exception:
        logger.warning("LLM parse failed for %s; retrying with strict prompt", meta.video_id)
        raw2 = _call_llm(prompt + _STRICT_SUFFIX, cfg)
        try:
            return _parse_response(raw2)
        except Exception as exc:
            raise PipelineError(f"LLM extraction failed for {meta.video_id}: could not parse JSON") from exc


_OVERVIEW_PROMPT = """\
You are a knowledge synthesis assistant. Given summaries of several videos in a
learning series, write a cohesive 2-4 paragraph outline that synthesizes the key
themes and progression of ideas. Be informative and avoid generic filler.

Videos:
{videos}

Respond with only the outline text (no JSON, no headers).
"""


def generate_overview(
    list_videos: list[dict],
    cfg: AIConfig,
    output_language: str = "ui",
    ui_lang: str | None = None,
) -> str:
    """Generate an overview outline from a list of {title, summary} dicts."""
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    videos_text = "\n\n".join(f"### {v['title']}\n{v['summary']}" for v in list_videos)
    prompt = lang_instruction + "\n\n" + _OVERVIEW_PROMPT.format(videos=videos_text)
    return _call_llm(prompt, cfg).strip()
