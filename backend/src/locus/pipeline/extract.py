"""LLM knowledge extraction step."""

import json
import logging

import httpx
from pydantic import BaseModel

from locus.adapters.base import VideoMeta
from locus.config import AIConfig
from locus.pipeline.audio import PipelineError

logger = logging.getLogger(__name__)

_TRANSCRIPT_TOKEN_WARN = 100_000
_TRANSCRIPT_CHAR_LIMIT = 400_000  # ~100K tokens at ~4 chars/token


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

_STRICT_SUFFIX = "\nReturn ONLY valid JSON. Do not add any explanation or markdown fences."


def _call_llm(prompt: str, cfg: AIConfig) -> str:
    """Dispatch to the appropriate LLM provider. Returns raw text response."""
    if cfg.provider == "anthropic":
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=cfg.api_key)
        msg = client.messages.create(
            model=cfg.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    if cfg.provider == "openai":
        import openai  # noqa: PLC0415

        client = openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content

    # ollama / custom: plain HTTP
    base = cfg.base_url or "http://localhost:11434"
    r = httpx.post(
        f"{base}/api/chat",
        json={
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _parse_response(text: str) -> ExtractionResult:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return ExtractionResult.model_validate(json.loads(text))


def extract_knowledge(
    transcript_text: str,
    meta: VideoMeta,
    cfg: AIConfig,
) -> ExtractionResult:
    if len(transcript_text) > _TRANSCRIPT_CHAR_LIMIT:
        logger.warning("Transcript for %s exceeds ~100K tokens; truncating", meta.video_id)
        transcript_text = transcript_text[:_TRANSCRIPT_CHAR_LIMIT]

    prompt = _EXTRACTION_PROMPT.format(title=meta.title, transcript=transcript_text)

    raw = _call_llm(prompt, cfg)
    try:
        return _parse_response(raw)
    except Exception:
        logger.warning("LLM parse failed for %s; retrying with strict prompt", meta.video_id)
        raw2 = _call_llm(prompt + _STRICT_SUFFIX, cfg)
        try:
            return _parse_response(raw2)
        except Exception as exc:
            raise PipelineError(
                f"LLM extraction failed for {meta.video_id}: could not parse JSON"
            ) from exc


_OVERVIEW_PROMPT = """\
You are a knowledge synthesis assistant. Given summaries of several videos in a
learning series, write a cohesive 2-4 paragraph outline that synthesizes the key
themes and progression of ideas. Be informative and avoid generic filler.

Videos:
{videos}

Respond with only the outline text (no JSON, no headers).
"""


def generate_overview(list_videos: list[dict], cfg: AIConfig) -> str:
    """Generate an overview outline from a list of {title, summary} dicts."""
    videos_text = "\n\n".join(f"### {v['title']}\n{v['summary']}" for v in list_videos)
    prompt = _OVERVIEW_PROMPT.format(videos=videos_text)
    return _call_llm(prompt, cfg).strip()
