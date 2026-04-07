"""LLM knowledge synthesis step."""

import logging

from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _LANG_INSTRUCTION,
    _call_llm,
    _resolved_lang,
)

logger = logging.getLogger(__name__)


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
    llm_timeout: int = 120,
    llm_max_tokens: int = 2048,
) -> str:
    """Generate an overview outline from a list of {title, summary} dicts."""
    lang = _resolved_lang(output_language, ui_lang)
    lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
    videos_text = "\n\n".join(f"### {v['title']}\n{v['summary']}" for v in list_videos)
    prompt = lang_instruction + "\n\n" + _OVERVIEW_PROMPT.format(videos=videos_text)
    return _call_llm(
        prompt,
        cfg,
        llm_timeout=llm_timeout,
        llm_max_tokens=llm_max_tokens,
    ).strip()
