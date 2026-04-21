"""Shared helpers used across pipeline stages."""

import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, TypeVar

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

from bibilab.config import AIConfig

logger = logging.getLogger(__name__)

# Module-level client cache: key -> client instance
_client_cache: dict[tuple[str, str, str, str | None], object] = {}

# Async client cache: key -> async client instance
_async_client_cache: dict[tuple[str, str, str, str | None], object] = {}

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
    cache_key = (cfg.protocol, cfg.api_key, cfg.model, cfg.base_url)

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
        return msg.content[0].text

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
    return resp.choices[0].message.content


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
    llm_max_tokens: int = 2048,
    system: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    cache_key = (cfg.protocol, cfg.api_key, cfg.model, cfg.base_url)

    if cfg.protocol == "anthropic":
        if cache_key not in _async_client_cache:
            _async_client_cache[cache_key] = AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
        client: AsyncAnthropic = _async_client_cache[cache_key]

        kwargs = dict(
            model=cfg.model,
            max_tokens=llm_max_tokens,
            messages=messages,
            system=system,
            timeout=llm_timeout,
        )
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

        response = await client.chat.completions.create(
            model=cfg.model,
            messages=full_messages,
            tools=tool_params,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
            stream=True,
        )

        pending: dict[int, dict] = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]

            if choice.delta.content:
                for info in pending.values():
                    args_str = info["args_str"]
                    if args_str:
                        try:
                            info["arguments"] = json.loads(args_str)
                        except json.JSONDecodeError:
                            pass
                    if info["name"]:
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=info["id"],
                                name=info["name"],
                                arguments=info.get("arguments", {}),
                            ),
                        )
                pending.clear()
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
                    pass
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
