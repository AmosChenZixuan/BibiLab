"""Shared helpers used across pipeline stages."""

import json
import logging
from typing import TypeVar

import anthropic
import openai
from pydantic import BaseModel

from bibilab.config import AIConfig

logger = logging.getLogger(__name__)

# Module-level client cache: key -> client instance
_client_cache: dict[tuple[str, str, str], object] = {}

_LANG_INSTRUCTION = {
    "en": "Respond in English only. Do not use any other language.",
    "zh": "请用中文回答。不要使用其他语言。",
}

_LANG_NAME = {
    "en": "English",
    "zh": "Chinese",
}


def _resolved_lang(output_language: str, ui_lang: str | None) -> str:
    if output_language == "ui":
        return ui_lang or "en"
    return output_language


_STRICT_SUFFIX = "\nReturn ONLY valid JSON. Do not add any explanation or markdown fences."


def _call_llm(
    prompt: str,
    cfg: AIConfig,
    llm_timeout: int = 120,
    llm_max_tokens: int = 2048,
) -> str:
    """Dispatch to the appropriate LLM protocol. Returns raw text response."""
    cache_key = (cfg.protocol, cfg.api_key, cfg.base_url)

    if cfg.protocol == "anthropic":
        if cache_key not in _client_cache:
            _client_cache[cache_key] = anthropic.Anthropic(api_key=cfg.api_key)
        client: anthropic.Anthropic = _client_cache[cache_key]
        msg = client.messages.create(
            model=cfg.model,
            max_tokens=llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    # openai (includes OpenAI-compatible providers via base_url)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
    client: openai.OpenAI = _client_cache[cache_key]
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=llm_max_tokens,
    )
    return resp.choices[0].message.content


_T = TypeVar("_T", bound=BaseModel)


def _parse_llm_json_response(text: str, model_cls: type[_T]) -> _T:
    """Strip markdown fences and parse LLM JSON response into a Pydantic model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return model_cls.model_validate(json.loads(text))
