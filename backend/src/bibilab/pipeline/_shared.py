"""Shared helpers used across pipeline stages."""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Callable, TypeVar

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

from bibilab.config import AIConfig

if TYPE_CHECKING:
    from bibilab.adapters.base import VideoMeta

logger = logging.getLogger(__name__)

# Shared cl100k_base token estimator. It's OpenAI's tokenizer — an approximation
# for other providers, but its drift only matters when input nears the window,
# which _INPUT_MARGIN absorbs. Used here for the input budget and by the chunker
# (pipeline/chunk.py) for chunk sizing.
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate token count with the shared cl100k_base encoder."""
    return len(_enc.encode(text))


def format_hms(seconds: float | None) -> str:
    """Format seconds as M:SS (or H:MM:SS when >= 1 hour). `None` → "0:00".
    Shared between the section fence header (chat_tools) and the per-segment
    timestamp in format_turns (transcribe) — both call sites must produce
    the same shape or the rendered body has inconsistent timestamps.
    """
    s = int(seconds or 0)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def interpreting_providers() -> list[str]:
    """ONNX kernel-based execution providers — no compiler-based EP.

    ONNX Runtime splits EPs into two designs: *kernel-based* ones dispatch
    stateless per-op kernels (CPU, CUDA, ROCm), while *compiler-based* ones run
    GetCapability/Compile() to fuse partitions into a JIT'd custom op (CoreML,
    DirectML, TensorRT, OpenVINO, MIGraphX, NNAPI, QNN, …). The JIT holds a
    second copy of the weights and recompiles per input shape — pathological for
    this BERT-family embedding model. CoreML, measured:

    - Chat query (one short string/turn): ~5 ms on CPU vs ~37 ms under CoreML,
      and the CoreML session inflates to ~3.4 GB vs ~1.0 GB on CPU.
    - Ingest (batches of 512-token chunks through the shared singleton session):
      CoreML recompiles per input shape and collapses — ~48x slower (measured
      0.2 vs ~10 chunks/s), OOMs at a full source's batch, and deadlocks when
      two ingest jobs hit the session concurrently. CPU stays fast and stable.

    There is no speed case for a compiling accelerator here, so allowlist the
    kernel-based EPs (honest memory). The allowlist, not a denylist, is the
    fail-safe direction: the compiler-based set is large and keeps growing, and
    an unrecognised new one must fall back to CPU/CUDA, never slip through to a
    fresh OOM. ROCm (AMD's HIP port of the CUDA kernels) is kernel-based, so it
    stays alongside CUDA.
    """
    import onnxruntime as ort  # noqa: PLC0415

    return [
        p
        for p in ort.get_available_providers()
        if p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider")
    ]


# Input-side margin. Absorbs tokenizer drift (cl100k vs. provider-native) and
# per-message framing overhead. Single-knob; no per-tier ceiling.
_INPUT_MARGIN = 2048

# stop_reason (Anthropic) / finish_reason (OpenAI) values that mean the model
# was cut off by the output token limit. Only these justify a "raise max output
# tokens" hint — any other no-text ending is an empty response, not a budget cap.
_LENGTH_STOP_REASONS = frozenset({"max_tokens", "length"})


class ContextWindowExceededError(ValueError):
    """Input alone exceeds the configured context window, so no valid max_tokens
    exists. Raised by resolve_max_tokens for every LLM call site; chat maps it to
    an i18n error code, pipeline stages surface its message as the job error."""


class LLMOutputBudgetExceededError(ValueError):
    """LLM returned no text AND was cut off by the output token limit
    (stop_reason/finish_reason in _LENGTH_STOP_REASONS) — thinking or generation
    exhausted the budget. Subclasses ValueError so any legacy `except ValueError`
    continues to catch it. Mapped to llm_output_budget_exceeded in chat router
    so the frontend can surface a 'raise max output tokens' hint."""


class LLMEmptyResponseError(ValueError):
    """LLM returned no text for a reason *other* than the token limit (normal
    stop with empty body, a content refusal that yielded nothing, an unknown
    terminal state). Distinct from LLMOutputBudgetExceededError so the frontend
    does NOT tell the user to raise max output tokens — that advice is false
    here. Mapped to llm_empty_response. Often transient, so pipeline retry
    loops (digest) deliberately do NOT re-raise it immediately."""


def _no_text_error(reason: str | None, max_tokens: int) -> ValueError:
    """Pick the right no-text error from the terminal stop/finish reason.

    Length cutoff → budget error (raise max output tokens). Anything else
    (including an unknown/missing reason) → empty-response error, so we never
    give false budget advice for a refusal or transient blank."""
    if reason in _LENGTH_STOP_REASONS:
        return LLMOutputBudgetExceededError(
            f"LLM hit the output token limit with no text (reason={reason}, max_tokens={max_tokens}). "
            "Thinking or generation may have consumed the whole budget."
        )
    return LLMEmptyResponseError(f"LLM returned no text content (reason={reason}, max_tokens={max_tokens}).")


def resolve_max_tokens(cfg: AIConfig, input_text: str) -> int:
    """Return user-chosen max output tokens. Raise if input + output + margin > context_window.

    The output budget is an explicit user setting (cfg.max_output_tokens),
    not derived from context_window. Single overflow check; no slow path,
    no dynamic ceiling, no anti-correlation. Chat and pipeline callers
    re-raise ContextWindowExceededError; structured callers (digest, artifact)
    treat it as non-retryable.
    """
    estimated_input = count_tokens(input_text)
    if estimated_input + cfg.max_output_tokens + _INPUT_MARGIN > cfg.context_window:
        raise ContextWindowExceededError(
            f"Input (~{estimated_input} tokens) + max output "
            f"({cfg.max_output_tokens} tokens) exceeds the "
            f"{cfg.context_window}-token context window."
        )
    return cfg.max_output_tokens


def _serialize_messages(messages: list[dict], system: str | None, tools: "list[ToolDefinition] | None" = None) -> str:
    """Flatten system prompt + chat messages + tool schemas into one string for
    token estimation. Tool definitions are the largest fixed input block in a
    tool-calling turn, so they're counted, not approximated by the margin.

    Content is usually a plain string; non-string content (tool blocks) is
    JSON-dumped. OpenAI tool-calling turns also carry `tool_calls` (assistant),
    `tool_call_id` + `name` (tool) on the message envelope — without these the
    function-name and arguments JSON, often hundreds of bytes, are invisible
    to the estimate and the resolved max_tokens is over-allocated.
    """
    parts = [system] if system else []
    for m in messages:
        content = m.get("content")
        parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
        for key in ("tool_calls", "tool_call_id", "name"):
            if m.get(key) is not None:
                v = m[key]
                parts.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
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
# not reliably map either to the intended language.
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


def resolve_response_language(cfg: AIConfig, ui_lang: str | None) -> str:
    """Return the language string to use in LLM-generated output.

    AIConfig.output_language wins when explicitly set; "ui" means follow
    the UI's X-UI-Lang header (passed in as ui_lang). Falls back to "en"
    when "ui" is selected but no ui_lang was provided, so the resolver
    never returns None/empty for downstream prompt interpolation.
    """
    if cfg.output_language == "ui":
        return ui_lang or "en"
    return cfg.output_language


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
            raise _no_text_error(getattr(msg, "stop_reason", None), llm_max_tokens)
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
        raise _no_text_error(getattr(resp.choices[0], "finish_reason", None), llm_max_tokens)
    return content


_T = TypeVar("_T", bound=BaseModel)


def _parse_llm_json_response(text: str, model_cls: type[_T]) -> _T:
    """Strip markdown fences and parse LLM JSON response into a Pydantic model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return model_cls.model_validate(json.loads(text))


# SDK exception → frontend-stable i18n error code. Order matters: the first
# matching type wins, so concrete subclasses (RateLimitError, AuthError) must
# appear before their base (APIError). Used by _classify_llm_error.
_ERROR_CODE_MAP: tuple[tuple[type[Exception], str], ...] = (
    (ContextWindowExceededError, "llm_context_window_exceeded"),
    (LLMOutputBudgetExceededError, "llm_output_budget_exceeded"),
    (LLMEmptyResponseError, "llm_empty_response"),
    (openai.APIConnectionError, "llm_connection_error"),
    (anthropic.APIConnectionError, "llm_connection_error"),
    (openai.AuthenticationError, "llm_auth_error"),
    (openai.PermissionDeniedError, "llm_auth_error"),
    (anthropic.AuthenticationError, "llm_auth_error"),
    (openai.RateLimitError, "llm_rate_limit_error"),
    (anthropic.RateLimitError, "llm_rate_limit_error"),
    (openai.APIError, "llm_api_error"),
    (anthropic.APIError, "llm_api_error"),
)


def _classify_llm_error(exception: Exception) -> str:
    """Map SDK exception types to a stable error code for i18n on the frontend."""
    for exc_type, code in _ERROR_CODE_MAP:
        if isinstance(exception, exc_type):
            return code
    return "internal_error"


# Sync helper. Async callers (worker) reach it via asyncio.to_thread.
# ``max_attempts`` defaults to ``len(prompts)`` — digest passes
# [plain, strict, strict] for 3 distinct prompts (default behavior); worker
# passes [single_prompt] with max_attempts=3 to retry the same prompt 3×.
_T2 = TypeVar("_T2")


def _call_llm_with_retry(
    prompts: list[str],
    parse_fn: Callable[[str], _T2],
    *,
    cfg: AIConfig,
    label: str,
    llm_timeout: int = 120,
    meta: "VideoMeta | None" = None,
    max_attempts: int | None = None,
) -> _T2:
    """Cycle through ``prompts`` retrying transient failures up to ``max_attempts`` total.

    On ``ContextWindowExceededError`` / ``LLMOutputBudgetExceededError`` raise
    immediately (input overflow and output budget exhaustion are identical
    across attempts). On any other exception, log + retry. On exhaustion
    raise ``PipelineError``. ``meta`` is optional — when provided, the
    video_id is included in retry logs."""
    # Local import: pipeline/audio imports from pipeline/_shared transitively,
    # so a top-level import here would cycle.
    from bibilab.pipeline.audio import PipelineError

    attempts_budget = max_attempts if max_attempts is not None else len(prompts)
    sequence = [prompts[i % len(prompts)] for i in range(attempts_budget)]
    last_exc: Exception | None = None
    for attempt, p in enumerate(sequence, start=1):
        try:
            raw = _call_llm(p, cfg, llm_timeout=llm_timeout)
            return parse_fn(raw)
        except (ContextWindowExceededError, LLMOutputBudgetExceededError):
            # Input overflow and output budget exhaustion are identical
            # across attempts (same prompt + same budget) — re-raise rather
            # than burning two more calls. Transient errors fall through
            # to the generic except below and get retried.
            raise
        except Exception as exc:
            last_exc = exc
            if meta is not None:
                logger.warning(
                    "LLM %s failed for %s (attempt %d/%d): %s",
                    label,
                    meta.video_id,
                    attempt,
                    attempts_budget,
                    exc,
                )
            else:
                logger.warning("LLM %s failed (attempt %d/%d): %s", label, attempt, attempts_budget, exc)

    if meta is not None:
        error_msg = f"LLM {label} exhausted all retries for {meta.video_id}: {last_exc}"
    else:
        error_msg = f"LLM {label} exhausted all retries: {last_exc}"
    logger.error(error_msg)
    raise PipelineError(error_msg) from last_exc


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
    # Terminal stop_reason (Anthropic) / finish_reason (OpenAI). Set only on the
    # "done" event so callers can tell a length cutoff from a normal empty end.
    stop_reason: str | None = None


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

        stop_reason: str | None = None
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
                elif event.type == "message_delta":
                    stop_reason = getattr(event.delta, "stop_reason", None) or stop_reason
        yield StreamEvent(type="done", stop_reason=stop_reason)
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

        finish_reason: str | None = None
        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

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

        yield StreamEvent(type="done", stop_reason=finish_reason)
