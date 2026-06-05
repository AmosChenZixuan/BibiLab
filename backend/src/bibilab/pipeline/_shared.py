"""Shared helpers used across pipeline stages."""

import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, TypeVar

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

from bibilab.config import AIConfig

logger = logging.getLogger(__name__)

# Shared cl100k_base token estimator. It's OpenAI's tokenizer — an approximation
# for other providers, but its drift only matters when input nears the window,
# which _INPUT_MARGIN absorbs. Used here for the input budget and by the chunker
# (pipeline/chunk.py) for chunk sizing.
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate token count with the shared cl100k_base encoder."""
    return len(_enc.encode(text))


# Output budget bounds. The ceiling is a generous fixed cap that comfortably
# fits extended thinking + a full answer for every role (chat, digest, artifact,
# summary) — large models routinely spend 10K+ tokens thinking. The margin
# absorbs tokenizer drift (we estimate with cl100k; providers use their own)
# and per-message framing overhead.
_OUTPUT_CEILING = 32768
_INPUT_MARGIN = 2048


def resolve_max_tokens(cfg: AIConfig, input_text: str) -> int:
    """Auto-scale the per-call output budget from the configured context window.

    max_tokens = min(context_window - estimated_input - margin, ceiling)

    The precise token count only changes the result when input nears the window
    (otherwise the ceiling binds), so we skip it in the common case: cl100k
    tokens never exceed UTF-8 byte length, so if the bytes already fit under the
    ceiling threshold the ceiling wins without encoding (#432).

    When input alone overflows the window no max_tokens is valid — raise, since
    the only fix is bounding how much read_source injects, which is backend-side.
    """
    if len(input_text.encode("utf-8")) <= cfg.context_window - _INPUT_MARGIN - _OUTPUT_CEILING:
        return _OUTPUT_CEILING
    estimated_input = count_tokens(input_text)
    available = cfg.context_window - estimated_input - _INPUT_MARGIN
    if available <= 0:
        raise ValueError(
            f"Input (~{estimated_input} tokens) exceeds the {cfg.context_window}-token "
            "context window — read_source pulled more than fits this turn."
        )
    return min(available, _OUTPUT_CEILING)


def _serialize_messages(messages: list[dict], system: str | None, tools: "list[ToolDefinition] | None" = None) -> str:
    """Flatten system prompt + chat messages + tool schemas into one string for
    token estimation. Tool definitions are the largest fixed input block in a
    tool-calling turn, so they're counted, not approximated by the margin.

    Content is usually a plain string; non-string content (tool blocks) is
    JSON-dumped.
    """
    parts = [system] if system else []
    for m in messages:
        content = m.get("content")
        parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    for t in tools or []:
        parts.append(f"{t.name}\n{t.description}\n{json.dumps(t.parameters, ensure_ascii=False)}")
    return "\n".join(parts)


# Module-level client cache: key -> client instance
_client_cache: dict[tuple[str, str, str, str | None], object] = {}

# Async client cache: key -> async client instance
_async_client_cache: dict[tuple[str, str, str, str | None], object] = {}

_LANG_INSTRUCTION = {
    "en": "Respond in English only. Do not use any other language.",
    "zh": "请用中文回答。不要使用其他语言。",
}

# Native-language short name (e.g. "简体中文") for LLM-prompt interpolation.
# Used as a stronger, language-specific instruction than the bare ISO code
# ("zh") or the English name ("Chinese") — smaller / less capable models may
# not reliably map either to the intended language. #402.
_LANG_NATIVE_NAME: dict[str, str] = {
    "en": "English",
    "zh": "简体中文",
}


def _lang_output_directive(lang: str) -> str:
    """Return the "All output fields MUST be written in X." suffix for the user's
    language. Used by digest and artifact prompts to reinforce the longer
    _LANG_INSTRUCTION. Falls back to English for unknown codes — matches the
    _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"]) convention so the LLM
    always gets a coherent English directive, never a raw ISO code."""
    return f"All output fields MUST be written in {_LANG_NATIVE_NAME.get(lang, 'English')}."


def _resolved_lang(output_language: str, ui_lang: str | None) -> str:
    if output_language == "ui":
        return ui_lang or "en"
    return output_language


_STRICT_SUFFIX = "\nReturn ONLY valid JSON. Do not add any explanation or markdown fences."


def _call_llm(
    prompt: str,
    cfg: AIConfig,
    llm_timeout: int = 120,
) -> str:
    """Dispatch to the appropriate LLM protocol. Returns raw text response."""
    cache_key = (cfg.protocol, cfg.api_key, cfg.model, cfg.base_url)
    llm_max_tokens = resolve_max_tokens(cfg, prompt)

    if cfg.protocol == "anthropic":
        if cache_key not in _client_cache:
            _client_cache[cache_key] = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
        client: anthropic.Anthropic = _client_cache[cache_key]
        msg = client.messages.create(
            model=cfg.model,
            max_tokens=llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=llm_timeout,
        )
        text_block = next((block for block in msg.content if block.type == "text"), None)
        if text_block is None:
            stop_reason = getattr(msg, "stop_reason", "unknown")
            raise ValueError(
                f"LLM returned no text content (stop_reason={stop_reason}, max_tokens={llm_max_tokens}). "
                "This may happen when thinking consumes all output tokens."
            )
        return text_block.text

    # openai (includes OpenAI-compatible providers via base_url)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
    client: openai.OpenAI = _client_cache[cache_key]
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=llm_max_tokens,
        timeout=llm_timeout,
    )
    content = resp.choices[0].message.content
    if not content:
        finish_reason = getattr(resp.choices[0], "finish_reason", "unknown")
        raise ValueError(
            f"LLM returned no text content (finish_reason={finish_reason}, max_tokens={llm_max_tokens}). "
            "This may happen when reasoning consumes all output tokens."
        )
    return content


_T = TypeVar("_T", bound=BaseModel)


def _parse_llm_json_response(text: str, model_cls: type[_T]) -> _T:
    """Strip markdown fences and parse LLM JSON response into a Pydantic model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return model_cls.model_validate(json.loads(text))


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class StreamEvent:
    type: str
    content: str | None = None
    tool_call: ToolCall | None = None


def _to_openai_tool(tool: ToolDefinition) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _to_anthropic_tool(tool: ToolDefinition) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


async def stream_llm(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition] | None = None,
    llm_timeout: int = 120,
    system: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    cache_key = (cfg.protocol, cfg.api_key, cfg.model, cfg.base_url)
    llm_max_tokens = resolve_max_tokens(cfg, _serialize_messages(messages, system, tools))

    if cfg.protocol == "anthropic":
        if cache_key not in _async_client_cache:
            _async_client_cache[cache_key] = AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
        client: AsyncAnthropic = _async_client_cache[cache_key]

        kwargs = dict(
            model=cfg.model,
            max_tokens=llm_max_tokens,
            messages=messages,
            timeout=llm_timeout,
        )
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamEvent(type="delta", content=event.delta.text)
                elif event.type == "content_block_stop":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input,
                            ),
                        )
        yield StreamEvent(type="done")
    else:
        if cache_key not in _async_client_cache:
            _async_client_cache[cache_key] = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
        client: AsyncOpenAI = _async_client_cache[cache_key]

        tool_params = [_to_openai_tool(t) for t in tools] if tools else None

        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}, *messages]

        kwargs = {
            "model": cfg.model,
            "messages": full_messages,
            "max_tokens": llm_max_tokens,
            "timeout": llm_timeout,
            "stream": True,
        }
        if tool_params is not None:
            kwargs["tools"] = tool_params

        response = await client.chat.completions.create(**kwargs)

        pending: dict[int, dict] = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]

            if choice.delta.content:
                yield StreamEvent(type="delta", content=choice.delta.content)

            if choice.delta.tool_calls:
                for tc_delta in choice.delta.tool_calls:
                    idx = tc_delta.index
                    info = pending.setdefault(idx, {"id": "", "name": "", "args_str": "", "arguments": {}})
                    if tc_delta.id:
                        info["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            info["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            info["args_str"] += tc_delta.function.arguments

        for info in pending.values():
            args_str = info["args_str"]
            if args_str:
                try:
                    info["arguments"] = json.loads(args_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call arguments: %s", args_str)
                    continue
            if info["name"]:
                yield StreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=info["id"],
                        name=info["name"],
                        arguments=info.get("arguments", {}),
                    ),
                )

        yield StreamEvent(type="done")
