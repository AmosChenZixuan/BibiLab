# 228 — Tool-Calling Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 8K-token classifier LLM call with an LLM tool that dispatches retrieval mid-stream (25× token savings). Delete `route.py` and all ChatMode/QueryType plumbing.

**Architecture:** `stream_with_tools` wraps `stream_llm` in a bounded loop. The LLM decides whether to call `retrieve` (loopback — result feeds back for another turn) or `generate_report` (terminal — exits). `stream_llm` already passes messages to the underlying SDKs as-is, so content-block support is a verification step, not a refactor. The bulk of the work is deletion: route.py, ChatMode, QueryType, query_routing_enabled, query_classifications table, conversations.mode column, PATCH endpoint, and frontend ChatMode types.

**Tech Stack:** Python/FastAPI backend, React + TypeScript frontend, SQLite, Anthropic + OpenAI SDKs

**Spec:** `docs/specs/228_tool_calling_retrieval.md`

---

### Task 1: Verify content-block passthrough in stream_llm

**Files:**
- Create: `backend/tests/test_stream_with_tools.py`
- Read: `backend/src/bibilab/pipeline/_shared.py:144-250`

- [ ] **Step 1: Write the test file for content-block passthrough**

```python
"""Tests for stream_with_tools loop and content-block passthrough."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bibilab.pipeline._shared import stream_llm


def an_async_generator(items):
    async def gen():
        for item in items:
            yield item
    return gen()


@pytest.mark.asyncio
async def test_stream_llm_passes_list_content_to_anthropic():
    """Anthropic path: messages with list-type content are passed through as-is."""
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="anthropic", model="claude", api_key="test", base_url=None)
    captured = []

    class FakeStream:
        def __init__(self, **kwargs):
            captured.append(kwargs["messages"])
        async def __aenter__(self):
            return an_async_generator([])
        async def __aexit__(self, *args):
            pass

    with patch("bibilab.pipeline._shared.AsyncAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock(messages=MagicMock(stream=FakeStream))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "retrieve", "input": {"query": "test", "search_mode": "factual"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"result":"ok"}'}]},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg, llm_max_tokens=512):
            pass

    assert len(captured) == 1
    assert isinstance(captured[0][1]["content"], list)


@pytest.mark.asyncio
async def test_stream_llm_passes_openai_tool_messages():
    """OpenAI path: tool_calls and role=tool messages pass through as-is."""
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url=None)
    captured = []

    async def fake_create(**kwargs):
        captured.append(kwargs["messages"])
        return an_async_generator([])

    with patch("bibilab.pipeline._shared.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock(chat=MagicMock(completions=MagicMock(create=fake_create)))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "retrieve", "arguments": '{"query":"test","search_mode":"factual"}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": '{"result":"ok"}'},
        ]
        async for _ in stream_llm(messages=messages, cfg=cfg, llm_max_tokens=512):
            pass

    assert len(captured) == 1
    msgs = captured[0]
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_call_id"] == "t1"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_stream_with_tools.py -v`
Expected: PASS — both paths already accept content-block messages.

- [ ] **Step 3: Verify by reading stream_llm code**

Confirm that `_shared.py:162` passes `messages=messages` directly to Anthropic SDK (which accepts list-type content natively), and `_shared.py:200` passes `messages=full_messages` directly to OpenAI SDK (which accepts `tool_calls` and `role: "tool"` natively).

---

### Task 2: Add RETRIEVE_TOOL, search_mode_to_params, and execute_retrieve to chat_tools.py

**Files:**
- Modify: `backend/src/bibilab/pipeline/chat_tools.py`
- Read: `backend/src/bibilab/pipeline/embed.py:24-27` (RetrievalParams)
- Read: `backend/src/bibilab/pipeline/embed.py:433-487` (retrieve function)

- [ ] **Step 1: Write tests for search_mode_to_params**

Create a new test file or extend existing:

Run: `uv run pytest tests/test_query_routing.py -v` first to see current `params_for_type` tests.

- [ ] **Step 2: Write unit tests for search_mode_to_params in a new test file**

Create `backend/tests/test_search_mode_params.py`:

```python
"""Tests for search_mode_to_params (replaces params_for_type)."""
import pytest
from bibilab.models._enums import RetrievalParams


def test_search_mode_to_params_factual():
    from bibilab.pipeline.chat_tools import search_mode_to_params
    params = search_mode_to_params("factual", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 4


def test_search_mode_to_params_breadth():
    from bibilab.pipeline.chat_tools import search_mode_to_params
    params = search_mode_to_params("breadth", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 10  # capped at sources_total


def test_search_mode_to_params_breadth_small_list():
    from bibilab.pipeline.chat_tools import search_mode_to_params
    params = search_mode_to_params("breadth", sources_total=2)
    assert params.depth_per_source == 1
    assert params.top_k == 4  # degrades to factual


def test_search_mode_to_params_analytical():
    from bibilab.pipeline.chat_tools import search_mode_to_params
    params = search_mode_to_params("analytical", sources_total=20)
    assert params.depth_per_source == 4
    assert params.top_k == 20  # min(sources_total, 12*3=36) = 20


def test_search_mode_to_params_unknown_falls_back_to_factual():
    from bibilab.pipeline.chat_tools import search_mode_to_params
    params = search_mode_to_params("garbage", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_search_mode_params.py -v`
Expected: FAIL — `search_mode_to_params` not defined.

- [ ] **Step 4: Add the code to chat_tools.py**

Add after `_VALID_ARTIFACT_TYPES` (after line 8):

```python
from bibilab.config import BibilabConfig
from bibilab.models._enums import RetrievalParams
from bibilab.pipeline.embed import retrieve

_PARAMS_BY_MODE = {
    "factual":    RetrievalParams(depth_per_source=1, top_k=4),
    "breadth":    RetrievalParams(depth_per_source=1, top_k=20),
    "analytical": RetrievalParams(depth_per_source=4, top_k=12),
}


def search_mode_to_params(search_mode: str, sources_total: int) -> RetrievalParams:
    if search_mode == "breadth" and sources_total < 3:
        return _PARAMS_BY_MODE["factual"]
    base = _PARAMS_BY_MODE.get(search_mode, _PARAMS_BY_MODE["factual"])
    if search_mode == "breadth":
        return RetrievalParams(depth_per_source=base.depth_per_source, top_k=min(base.top_k, sources_total))
    if search_mode == "factual":
        return base
    top_k = max(base.top_k, min(sources_total, base.top_k * 3))
    return RetrievalParams(depth_per_source=base.depth_per_source, top_k=top_k)
```

Add after the existing `GENERATE_REPORT_TOOL` definition (before `execute_generate_report`):

```python
RETRIEVE_TOOL = ToolDefinition(
    name="retrieve",
    description=(
        "Retrieve information from video transcripts. Use when the user asks about "
        "video content, facts, comparisons, summaries, or anything requiring lookup "
        "across sources. Do NOT use for chitchat (thanks, greetings, rephrasing) or "
        "conversation-only queries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — key terms or question in the user's language",
            },
            "search_mode": {
                "type": "string",
                "enum": ["factual", "breadth", "analytical"],
                "description": (
                    "factual = specific fact from 1-2 sources; "
                    "breadth = survey/list across many sources; "
                    "analytical = comparison/analysis needing deep per-source coverage"
                ),
            },
        },
        "required": ["query", "search_mode"],
    },
)
```

Add after `execute_generate_report` (before `execute_tool`):

```python
async def execute_retrieve(
    query: str,
    search_mode: str,
    source_ids: list[str],
    cfg: BibilabConfig,
) -> dict:
    params = search_mode_to_params(search_mode, len(source_ids))
    result = await retrieve(query_text=query, source_ids=source_ids, cfg=cfg, params=params)
    return {
        "search_mode": search_mode,
        "candidates_evaluated": result.candidates_evaluated,
        "sources_with_hits": result.sources_with_hits,
        "sources_total": result.sources_total,
        "source_coverage": [
            {"video_id": s.video_id, "title": s.video_title}
            for s in result.source_coverage
        ],
        "_chunks": [
            {"title": c.video_title, "start": c.timestamp_start, "end": c.timestamp_end, "content": c.content}
            for c in result.chunks
        ],
    }
```

- [ ] **Step 5: Update execute_tool to dispatch to execute_retrieve**

Modify `execute_tool` in `chat_tools.py:61-82` to add retrieve dispatch before the generate_report check:

```python
async def execute_tool(
    tool_name: str,
    arguments: dict,
    list_id: str,
    source_ids: list[str],
    ui_lang: str,
    cfg: BibilabConfig,
) -> dict:
    if tool_name == "retrieve":
        return await execute_retrieve(
            query=arguments["query"],
            search_mode=arguments["search_mode"],
            source_ids=source_ids,
            cfg=cfg,
        )
    if tool_name == "generate_report":
        artifact_type = arguments.get("type")
        prompt = arguments.get("prompt")
        if not artifact_type or not prompt:
            raise ValueError("Missing required arguments: type and prompt")
        if artifact_type not in _VALID_ARTIFACT_TYPES:
            artifact_type = "custom_report"
        return await execute_generate_report(
            list_id=list_id,
            artifact_type=artifact_type,
            prompt=prompt,
            source_ids=source_ids,
            ui_lang=ui_lang,
        )
    raise ValueError(f"Unknown tool: {tool_name}")
```

Note: `execute_tool` signature gains `cfg: BibilabConfig` parameter. This changes callers in `chat.py`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_search_mode_params.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/src/bibilab/pipeline/chat_tools.py backend/tests/test_search_mode_params.py
git commit -m "feat | backend | add RETRIEVE_TOOL, search_mode_to_params, execute_retrieve (#228)"
```

---

### Task 3: Add stream_with_tools loop to chat.py

**Files:**
- Modify: `backend/src/bibilab/routers/chat.py`
- Read: `backend/src/bibilab/pipeline/_shared.py:104-250` (ToolDefinition, ToolCall, StreamEvent, stream_llm)

- [ ] **Step 1: Write tests for stream_with_tools**

Add to `backend/tests/test_stream_with_tools.py`:

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch

from bibilab.pipeline._shared import StreamEvent, ToolCall, ToolDefinition


@pytest.mark.asyncio
async def test_stream_with_tools_passthrough_no_tool_calls():
    """When LLM returns no tool_calls, events pass through and function returns."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url=None)

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="delta", content="Hello")
        yield StreamEvent(type="done")

    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        cfg=cfg,
        tools=[],
        execute_tool_fn=AsyncMock(),
    ):
        events.append(event)

    assert len(events) == 2
    assert events[0].type == "delta"
    assert events[1].type == "done"


@pytest.mark.asyncio
async def test_stream_with_tools_loopback_retrieve():
    """LLM calls retrieve → execute → feed result → second LLM turn."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url=None)

    retrieve_tool_call = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "search_mode": "factual"})

    call_count = 0
    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(type="tool_call", tool_call=retrieve_tool_call)
        else:
            yield StreamEvent(type="delta", content="Answer based on chunks")
            yield StreamEvent(type="done")

    async def fake_execute(tool_name, arguments):
        if tool_name == "retrieve":
            return {"search_mode": "factual", "candidates_evaluated": 5, "sources_with_hits": 2, "sources_total": 3, "source_coverage": [], "_chunks": []}
        raise ValueError(f"Unknown tool: {tool_name}")

    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "what is this about?"}],
        cfg=cfg,
        tools=[RETRIEVE_TOOL],
        execute_tool_fn=fake_execute,
    ):
        events.append(event)

    assert call_count == 2
    tool_result_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_result_events) == 1
    assert "factual" in tool_result_events[0].content
    delta_events = [e for e in events if e.type == "delta"]
    assert len(delta_events) == 1


@pytest.mark.asyncio
async def test_stream_with_tools_max_iterations():
    """Hard cap at MAX_TOOL_ITERATIONS prevents infinite loops."""
    from bibilab.config import AIConfig
    from bibilab.routers.chat import stream_with_tools, MAX_TOOL_ITERATIONS

    cfg = AIConfig(protocol="openai", model="gpt-4o", api_key="test", base_url=None)

    tc = ToolCall(id="c1", name="retrieve", arguments={"query": "test", "search_mode": "factual"})

    async def fake_stream(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="tool_call", tool_call=tc)

    events = []
    async for event in stream_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        cfg=cfg,
        tools=[RETRIEVE_TOOL],
        execute_tool_fn=AsyncMock(return_value={"ok": True}),
    ):
        events.append(event)

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "Max tool iterations" in error_events[0].content
```

Import at top of test file:
```python
from bibilab.pipeline.chat_tools import RETRIEVE_TOOL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stream_with_tools.py::test_stream_with_tools_passthrough_no_tool_calls -v`
Expected: FAIL — `stream_with_tools` not defined.

- [ ] **Step 3: Add stream_with_tools to chat.py**

Add these constants after `SSE_EVENT_RAG_META` (line 57):

```python
LOOPBACK_TOOLS = {"retrieve"}
MAX_TOOL_ITERATIONS = 3
```

Add `stream_with_tools` function before `chat_endpoint` (before line 147):

```python
async def stream_with_tools(
    messages: list[dict],
    cfg: AIConfig,
    tools: list[ToolDefinition],
    execute_tool_fn,
    system: str | None = None,
    llm_max_tokens: int = CHAT_MAX_TOKENS,
) -> AsyncGenerator[StreamEvent, None]:
    messages = list(messages)
    iteration = 0

    while True:
        iteration += 1
        tool_calls: list[ToolCall] = []
        async for event in stream_llm(messages, cfg, tools, system=system, llm_max_tokens=llm_max_tokens):
            if event.type == "tool_call":
                tool_calls.append(event.tool_call)
            else:
                yield event

        if not tool_calls:
            return

        if iteration > MAX_TOOL_ITERATIONS:
            yield StreamEvent(type="error", content="Max tool iterations exceeded")
            return

        results: dict[str, dict] = {}
        for tc in tool_calls:
            try:
                result = await execute_tool_fn(tc.name, tc.arguments)
                results[tc.id] = result
            except Exception as exc:
                yield StreamEvent(type="error", content=str(exc))
                return

            yield StreamEvent(
                type="tool_result",
                content=json.dumps({"name": tc.name, "result": _client_tool_result(tc.name, result)}),
            )

        if any(tc.name in LOOPBACK_TOOLS for tc in tool_calls):
            if cfg.protocol == "anthropic":
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                        for tc in tool_calls
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tc.id, "content": json.dumps(results[tc.id])}
                        for tc in tool_calls
                    ],
                })
            else:
                openai_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                messages.append({"role": "assistant", "content": None, "tool_calls": openai_tool_calls})
                for tc in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(results[tc.id]),
                    })
            continue

        return
```

Add the helper that strips `_chunks` from client-bound tool results:

```python
def _client_tool_result(name: str, result: dict) -> dict:
    """Strip internal fields before sending to client."""
    if name == "retrieve":
        return {k: v for k, v in result.items() if k != "_chunks"}
    return result
```

Add imports at top of `chat.py`:
```python
from bibilab.pipeline._shared import StreamEvent, ToolCall, ToolDefinition, stream_llm
from bibilab.pipeline.chat_tools import GENERATE_REPORT_TOOL, RETRIEVE_TOOL, execute_tool
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_stream_with_tools.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/bibilab/routers/chat.py backend/tests/test_stream_with_tools.py
git commit -m "feat | backend | add stream_with_tools loop with content-block support (#228)"
```

---

### Task 4: Rewire chat.py event_generator to use stream_with_tools

**Files:**
- Modify: `backend/src/bibilab/routers/chat.py` (event_generator in chat_endpoint)

- [ ] **Step 1: Update GROUNDING_SYSTEM_PROMPT**

Add rule 0 after `"CRITICAL RULES:\n"`:

```python
GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly based on the provided source material. "
    "CRITICAL RULES:\n"
    "0. If the user's question requires looking up video content (facts, comparisons, "
    "summaries, counts across sources), you MUST call the retrieve tool BEFORE answering. "
    "Do not answer from memory — always retrieve first for content questions. "
    "This does NOT apply when the user asks you to generate a report/artifact — "
    "the generate_report tool handles its own retrieval.\n"
    "1. ONLY use information from the source material provided below. Never use your own knowledge.\n"
    '2. If the excerpts do not contain the answer, say "The provided sources do not cover this topic."\n'
    "3. Quote or closely paraphrase the source material — do not reinterpret, editorialize, or add external context.\n"
    "4. Cite sources using EXACTLY this format: [video_title @ Ns-Ns] — e.g. [My Video @ 120s-145s].\n"
    "5. Use the generate_report tool when the user asks for summaries, study guides, blog posts, or custom reports.\n"
    "6. Do not ask follow-up questions, suggest next steps, or offer unsolicited advice.\n"
    "7. Be concise and direct. Answer in 1-3 sentences when possible."
)
```

- [ ] **Step 2: Rewrite event_generator in chat_endpoint**

Replace the body of `event_generator` (lines 226-309) with the new version. The key changes:
- Remove `classify_query` call + `log_query_classification`
- Remove pre-retrieval (`retrieve()` before `stream_llm`)
- Remove `rag_meta` event emission
- Build system message WITHOUT RAG context (the LLM fetches it via `retrieve` tool)
- Call `stream_with_tools` instead of `stream_llm`
- Execute tool in loop via `execute_tool` (updated to accept `cfg`)
- Handle retrieve tool_result → format RAG context → feed back (done inside stream_with_tools)
- Persist assistant message after loop completes

```python
async def event_generator():
    nonlocal first_response_deltas, tool_calls

    # Build system message (no pre-retrieval — LLM calls retrieve tool mid-stream)
    system_parts = [GROUNDING_SYSTEM_PROMPT]
    if existing_summary:
        system_parts.append(
            "Historical conversation summary (for context only — the current "
            "question may be about different sources than those summarized below):\n" + existing_summary
        )
    system_message = "\n\n".join(system_parts)

    messages_for_llm = history + [{"role": "user", "content": request.message}]

    async def execute_tool_bound(name: str, args: dict) -> dict:
        return await execute_tool(
            tool_name=name,
            arguments=args,
            list_id=list_id,
            source_ids=source_ids,
            ui_lang=ui_lang,
            cfg=cfg,
        )

    tools = [RETRIEVE_TOOL, GENERATE_REPORT_TOOL]

    try:
        async for event in stream_with_tools(
            messages=messages_for_llm,
            cfg=cfg.ai,
            tools=tools,
            execute_tool_fn=execute_tool_bound,
            system=system_message if system_message.strip() else None,
            llm_max_tokens=CHAT_MAX_TOKENS,
        ):
            if event.type == SSE_EVENT_DELTA:
                first_response_deltas.append(event.content or "")
                yield f"data: {json.dumps({'type': SSE_EVENT_DELTA, 'content': event.content})}\n\n"
            elif event.type == "tool_call":
                tool_calls.append(event.tool_call)
            elif event.type == "tool_result":
                yield f"data: {json.dumps({'type': SSE_EVENT_TOOL_RESULT, **json.loads(event.content)})}\n\n"
            elif event.type == SSE_EVENT_DONE:
                if not tool_calls:
                    yield f"data: {json.dumps({'type': SSE_EVENT_DONE})}\n\n"
            elif event.type == "error":
                logger.error("stream_with_tools error: %s", event.content)
                yield f"data: {json.dumps({'type': SSE_EVENT_ERROR, 'message': event.content})}\n\n"
                return
    except Exception:
        logger.exception("LLM streaming failed")
        await delete_messages_by_ids([user_msg_id])
        yield f"data: {json.dumps({'type': SSE_EVENT_ERROR, 'message': 'An internal error occurred'})}\n\n"
        return

    if not tool_calls:
        full_response = "".join(first_response_deltas)
        await create_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            metadata=None,
        )
        return

    # Collect coverage from retrieve tool_result and persist
    retrieve_result = None
    tool_call_meta = []
    for tc in tool_calls:
        if tc.name == "retrieve":
            try:
                result = await execute_tool_bound("retrieve", tc.arguments)
                retrieve_result = result
                tool_call_meta.append({"id": tc.id, "name": tc.name, "result": result})
            except Exception as exc:
                yield f"data: {json.dumps({'type': SSE_EVENT_ERROR, 'message': str(exc)})}\n\n"
                await create_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content="",
                    metadata={"tool_calls": [{"id": tc.id, "name": tc.name, "error": str(exc)}]},
                )
                return
        elif tc.name == "generate_report":
            tool_call_meta.append({"id": tc.id, "name": tc.name})

    meta: dict[str, Any] = {}
    if tool_call_meta:
        meta["tool_calls"] = tool_call_meta
    if retrieve_result:
        meta["rag"] = {
            "search_mode": retrieve_result.get("search_mode"),
            "candidates_evaluated": retrieve_result.get("candidates_evaluated"),
            "sources_with_hits": retrieve_result.get("sources_with_hits"),
            "sources_total": retrieve_result.get("sources_total"),
            "source_coverage": retrieve_result.get("source_coverage"),
        }

    await create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="",
        metadata=meta if meta else None,
    )
```

- [ ] **Step 3: Remove dead imports from chat.py**

Remove these imports (no longer used):
```python
from bibilab.db import (
    ...
    log_query_classification,  # DELETE
    ...
    update_conversation_mode,  # DELETE
)
from bibilab.models._enums import (
    QUERY_TYPE_FACTUAL,  # DELETE
    ChatMode,  # DELETE
    map_type_to_mode,  # DELETE
)
from bibilab.pipeline.route import classify_query, params_for_type  # DELETE
```

Also remove the `_format_rag_context`, `_format_chunk_line`, `_build_rag_payload` functions (lines 108-131) — no longer needed since retrieval happens inside tool execution, not pre-stream.

Remove `SSE_EVENT_RAG_META` constant (line 57).

- [ ] **Step 4: Run existing tests to see what breaks**

Run: `uv run pytest tests/test_chat_sse.py -v --tb=short`
Expected: Many failures — tests still reference `classify_query`, `rag_meta`, old flow.

- [ ] **Step 5: Commit**

```bash
git add backend/src/bibilab/routers/chat.py
git commit -m "feat | backend | rewire chat endpoint to use stream_with_tools (#228)"
```

---

### Task 5: Delete route.py, ChatMode, QueryType, and related dead code

**Files:**
- Delete: `backend/src/bibilab/pipeline/route.py` (entire file)
- Modify: `backend/src/bibilab/models/_enums.py`
- Modify: `backend/src/bibilab/config.py`
- Modify: `backend/src/bibilab/db.py`
- Modify: `backend/src/bibilab/models/chat.py`
- Delete: `backend/tests/test_query_routing.py` (entire file)

- [ ] **Step 1: Delete pipeline/route.py**

Run: `rm backend/src/bibilab/pipeline/route.py`

- [ ] **Step 2: Delete QueryType, ChatMode, map_type_to_mode from _enums.py**

Edit `backend/src/bibilab/models/_enums.py` — delete lines 13-37 (ChatMode, QueryType constants, map_type_to_mode). Keep `VideoStatus`, `RetrievalParams`. The file should contain only:

```python
from dataclasses import dataclass
from enum import Enum


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


@dataclass(frozen=True)
class RetrievalParams:
    depth_per_source: int
    top_k: int
```

- [ ] **Step 3: Delete query_routing_enabled from RagConfig**

Edit `backend/src/bibilab/config.py:84-91` — remove `query_routing_enabled` field:

```python
class RagConfig(BaseModel):
    max_distance: float = 0.8
    reranking_enabled: bool = True
    hybrid_enabled: bool = True
    rerank_min_score: float | None = None
```

- [ ] **Step 4: Delete log_query_classification, update_conversation_mode, query_classifications table from db.py**

In `backend/src/bibilab/db.py`:

Delete `_CREATE_QUERY_CLASSIFICATIONS` (lines 122-131).

Delete `log_query_classification` function (lines 862-876).

Delete `update_conversation_mode` function (lines 787-794).

In `bootstrap_db` (line 143), delete the line:
```python
await db.execute(_CREATE_QUERY_CLASSIFICATIONS)
```

In `_CREATE_CONVERSATIONS` (line 105), remove `mode` column:
```python
_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    list_id    TEXT NOT NULL UNIQUE REFERENCES lists(id) ON DELETE CASCADE,
    summary    TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
```

In `get_conversation_by_list` (line 616), change `SELECT id, list_id, summary, mode, created_at, updated_at` to `SELECT id, list_id, summary, created_at, updated_at`.

In `get_conversation` (line 781), same change: remove `mode` from SELECT.

Delete `ChatMode` import (line 14): `from bibilab.models._enums import ChatMode`.

- [ ] **Step 5: Delete PATCH endpoint and PatchConversationRequest**

In `backend/src/bibilab/routers/chat.py`, delete the `patch_conversation` endpoint (lines 97-105).

In `backend/src/bibilab/models/chat.py`:
- Delete `PatchConversationRequest` class (lines 64-65)
- Delete `ChatMode` import (line 6): `from bibilab.models._enums import ChatMode`
- In `ConversationResponse` (line 43), delete `mode: ChatMode` field
- In `ConversationResponse.from_row` (line 53), delete `mode=row["mode"]`

- [ ] **Step 6: Delete test_query_routing.py**

Run: `rm backend/tests/test_query_routing.py`

- [ ] **Step 7: Try running tests**

Run: `uv run pytest tests/test_chat_sse.py tests/test_chat_rag.py -v --tb=short 2>&1 | head -60`
Expected: Import errors from deleted modules — we'll fix these in Task 6.

- [ ] **Step 8: Verify no references to deleted modules**

Run:
```bash
grep -rn "pipeline.route\|classify_query\|params_for_type\|map_type_to_mode\|query_routing_enabled\|log_query_classification\|log_query_classification\|update_conversation_mode\|ChatMode\|CHAT_MODE\|QueryType\|QUERY_TYPE" backend/src/ --include="*.py" | grep -v __pycache__ | grep -v ".pyc"
```
Should only show hits in `chat_tools.py` (new `search_mode_to_params` uses `RetrievalParams`), `chat.py` (new code), and no dead references.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat | backend | delete route.py, ChatMode, QueryType, and classifier plumbing (#228)"
```

---

### Task 6: Update backend tests for new chat flow

**Files:**
- Modify: `backend/tests/test_chat_sse.py`
- Modify: `backend/tests/test_chat_rag.py`
- Read: `backend/tests/conftest.py` (for client fixture)

- [ ] **Step 1: Rewrite test_chat_sse.py tests**

The test file needs substantial changes:
- All tests that mock `classify_query` and `rag_meta` need rewriting
- New mock surface: `stream_with_tools` instead of `stream_llm`
- `rag_meta` event tests become `tool_result` event tests

Replace `backend/tests/test_chat_sse.py`:

```python
"""Tests for the SSE chat streaming endpoint (post-tool-calling refactor)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibilab.pipeline._shared import StreamEvent


def an_async_generator(items):
    async def gen():
        for item in items:
            yield item
    return gen()


@pytest.mark.asyncio
async def test_chat_endpoint_returns_sse_stream(client):
    """POST /lists/:id/chat returns text/event-stream with delta events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator([
            StreamEvent(type="delta", content="Hello"),
            StreamEvent(type="delta", content=" world"),
            StreamEvent(type="done"),
        ])
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_endpoint_404_for_missing_list(client):
    """Returns 404 when list does not exist."""
    resp = await client.post("/lists/nonexistent/chat", json={"message": "hi"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_endpoint_saves_user_message(client):
    """User message is persisted to DB after successful stream."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator([
            StreamEvent(type="delta", content="Hi"),
            StreamEvent(type="done"),
        ])
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hello"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    messages = conv_resp.json()["messages"]
    assert any(m["role"] == "user" and m["content"] == "hello" for m in messages)


@pytest.mark.asyncio
async def test_chat_endpoint_saves_assistant_message(client):
    """Assistant message is persisted to DB after stream completes."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator([
            StreamEvent(type="delta", content="Hello"),
            StreamEvent(type="done"),
        ])
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    messages = conv_resp.json()["messages"]
    assert any(m["role"] == "assistant" and m["content"] == "Hello" for m in messages)


@pytest.mark.asyncio
async def test_chat_endpoint_uses_conversation_history(client):
    """Prior conversation messages are included in LLM context."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    from bibilab.db import create_message, get_or_create_conversation

    conv_id = await get_or_create_conversation(list_id)
    await create_message(conv_id, "user", "Previous question", None)
    await create_message(conv_id, "assistant", "Previous answer", None)

    captured_messages = []

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        captured_messages.append(messages)
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", capture):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "follow-up"})

    assert resp.status_code == 200
    roles = [m["role"] for m in captured_messages[0]]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_chat_endpoint_includes_tools(client):
    """Both retrieve and generate_report tools are passed to stream_with_tools."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    captured_tools = []

    async def capture(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        captured_tools.append(tools)
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", capture):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    assert resp.status_code == 200
    tool_names = [t.name for t in captured_tools[0]]
    assert "retrieve" in tool_names
    assert "generate_report" in tool_names


@pytest.mark.asyncio
async def test_chat_endpoint_handles_generate_report_tool(client):
    """generate_report tool execution and SSE events."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        yield StreamEvent(type="delta", content="Generating...")
        result = await execute_tool_fn("generate_report", {"type": "brief", "prompt": "summarize"})
        yield StreamEvent(type="tool_result", content='{"name":"generate_report","result":{"artifact_id":"abc","name":"brief","type":"brief"}}')
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        resp = await client.post(f"/lists/{list_id}/chat", json={"message": "make a brief"})

    assert resp.status_code == 200
    assert "abc" in resp.text
    assert "tool_result" in resp.text


@pytest.mark.asyncio
async def test_retrieve_tool_result_has_coverage(client):
    """retrieve tool_result includes search_mode and coverage metadata."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    retrieve_result = {
        "search_mode": "factual",
        "candidates_evaluated": 30,
        "sources_with_hits": 1,
        "sources_total": 1,
        "source_coverage": [{"video_id": "bv1", "title": "Test Video"}],
    }

    async def fake_stream(messages, cfg, tools=None, execute_tool_fn=None, system=None, llm_max_tokens=2048):
        result = await execute_tool_fn("retrieve", {"query": "test", "search_mode": "factual"})
        yield StreamEvent(type="tool_result", content='{"name":"retrieve","result":' + __import__("json").dumps(result) + '}')
        yield StreamEvent(type="delta", content="Answer")
        yield StreamEvent(type="done")

    with patch("bibilab.routers.chat.stream_with_tools", fake_stream):
        with patch("bibilab.routers.chat.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = retrieve_result
            resp = await client.post(f"/lists/{list_id}/chat", json={"message": "what is this?"})

    assert resp.status_code == 200
    assert "tool_result" in resp.text
    assert "factual" in resp.text


@pytest.mark.asyncio
async def test_conversation_no_longer_has_mode(client):
    """After mode column dropped, conversation response has no mode field."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]

    with patch("bibilab.routers.chat.stream_with_tools") as mock:
        mock.return_value = an_async_generator([StreamEvent(type="done")])
        await client.post(f"/lists/{list_id}/chat", json={"message": "hi"})

    conv_resp = await client.get(f"/lists/{list_id}/conversation")
    assert "mode" not in conv_resp.json()["conversation"]


@pytest.mark.asyncio
async def test_patch_conversation_endpoint_gone(client):
    """PATCH /lists/:id/conversation returns 405 (no route)."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.patch(f"/lists/{list_id}/conversation", json={"mode": "broad"})
    assert resp.status_code == 405
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_chat_sse.py -v --tb=long`
Expected: Some failures — need to fix mock paths and imports.

- [ ] **Step 3: Fix test_chat_rag.py references**

In `backend/tests/test_chat_rag.py`, check for `from bibilab.pipeline.route import params_for_type` references (lines 991, 1007, 1016). These test the old `params_for_type`. Replace with `from bibilab.pipeline.chat_tools import search_mode_to_params` and update test assertions.

```python
# Line ~991: Replace
from bibilab.pipeline.route import params_for_type
fact = params_for_type(QUERY_TYPE_FACTUAL, sources_total=10)
# With
from bibilab.pipeline.chat_tools import search_mode_to_params
fact = search_mode_to_params("factual", sources_total=10)
```

Remove `QUERY_TYPE_*` imports from test_chat_rag.py.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v --tb=short 2>&1 | tail -30`
Expected: All tests pass (except possibly test_query_routing.py which was deleted — if pytest collects it, remove `__pycache__`).

- [ ] **Step 5: Commit**

```bash
git add backend/tests/
git commit -m "fix | backend | update tests for tool-calling retrieval (#228)"
```

---

### Task 7: Delete frontend ChatMode, rag_meta, and PATCH conversation

**Files:**
- Modify: `web/src/lib/constants.ts`
- Modify: `web/src/lib/chat-utils.ts`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/components/lists/hooks/useSSEStream.ts`
- Modify: `web/src/components/lists/ObsChip.tsx`
- Modify: `web/src/test/chat-mode-toggle.test.tsx`

- [ ] **Step 1: Update web/src/lib/constants.ts**

Delete ChatMode lines (1-4) and SSE_EVENT_RAG_META (line 11). Update SSEEventType:

```typescript
export const SSE_EVENT_DELTA = "delta" as const;
export const SSE_EVENT_DONE = "done" as const;
export const SSE_EVENT_ERROR = "error" as const;
export const SSE_EVENT_TOOL_RESULT = "tool_result" as const;
export const SSE_EVENT_CLEAR_TEXT = "clear_text" as const;
export type SSEEventType =
  | typeof SSE_EVENT_DELTA
  | typeof SSE_EVENT_DONE
  | typeof SSE_EVENT_ERROR
  | typeof SSE_EVENT_TOOL_RESULT
  | typeof SSE_EVENT_CLEAR_TEXT;
```

- [ ] **Step 2: Update web/src/lib/chat-utils.ts**

Change `RagMetadata`:
```typescript
export type RagSource = { video_id: string; title: string };
export type RagMetadata = {
  search_mode: string;
  candidates_evaluated: number;
  sources_with_hits: number;
  sources_total: number;
  source_coverage: RagSource[];
};
```

Delete `import type { ChatMode } from "@/lib/constants";` (line 1).

- [ ] **Step 3: Delete updateConversation from web/src/lib/api.ts**

Delete lines 302-307 (the `updateConversation` method in ConversationsClient).

Delete `import type { ChatMode } from "./constants";` (line 21).

Delete from the `ApiClient` interface (line 374):
```typescript
updateConversation(listId: string, patch: { mode: ChatMode }): Promise<Conversation | undefined>;
```

Delete from the factory (line 432):
```typescript
updateConversation: (listId, patch) => conversations.updateConversation(listId, patch),
```

- [ ] **Step 4: Update web/src/lib/types.ts**

Delete `import type { ChatMode } from "./constants";` (line 1).

In `Conversation` type (line 222), delete `mode: ChatMode;`:

```typescript
export type Conversation = {
  id: string;
  list_id: string;
  summary: string | null;
  created_at: string;
  updated_at: string;
};
```

- [ ] **Step 5: Update useSSEStream.ts**

In `web/src/components/lists/hooks/useSSEStream.ts`:

Delete `SSE_EVENT_RAG_META` import (line 12).

Update the tool_result handler (around line 146) to detect `retrieve` vs `generate_report`:

```typescript
} else if (event.type === SSE_EVENT_TOOL_RESULT) {
  const result = event.result as ToolResult;
  if (!result) return;
  if (result.name === "retrieve") {
    const rag = event.result as unknown as RagMetadata;
    updateAssistantMsg(assistantMsgId, { rag });
  } else {
    const toolCallData = { name: "generate_report", result };
    if (result.job_id && trackJobs) {
      trackJobs([{ id: result.job_id, producer: "artifact", label: result.type, contextKey: listId }]);
    }
    updateAssistantMsg(assistantMsgId, { toolCall: toolCallData });
  }
}
```

Delete the `SSE_EVENT_RAG_META` handler (lines 154-157):
```typescript
} else if (event.type === SSE_EVENT_RAG_META) {
  const rag = event.rag as RagMetadata;
  if (!rag) return;
  updateAssistantMsg(assistantMsgId, { rag });
```

The `RagMetadata` import now pulls in the updated type (search_mode instead of mode).

- [ ] **Step 6: Update ObsChip.tsx**

In `web/src/components/lists/ObsChip.tsx`, change `rag.mode` to `rag.search_mode` (line 39):

```typescript
<span className="font-medium text-ink">{rag.search_mode}</span>
```

- [ ] **Step 7: Update frontend tests**

In `web/src/test/chat-mode-toggle.test.tsx`:

Replace `"mode":"focused"` with `"search_mode":"factual"` in all SSE event strings. Replace `rag_meta` events with `tool_result` events:

```typescript
test("retrieve tool_result event attaches rag to in-progress message", async () => {
  vi.spyOn(window, "fetch").mockImplementation(() =>
    Promise.resolve(
      makeSseStream([
        'data: {"type":"tool_result","name":"retrieve","result":{"search_mode":"factual","candidates_evaluated":30,"sources_with_hits":1,"sources_total":1,"source_coverage":[{"video_id":"BV1test","title":"Test Video A"}]}}\n\n',
        'data: {"type":"delta","content":"Hello"}\n\n',
        'data: {"type":"done"}\n\n',
      ]),
    ),
  );
  // ... rest unchanged
});
```

Update the other two tests similarly.

Update the expand test assertion from `focused` to `factual`:
```typescript
expect(screen.getByText(/factual/i)).toBeInTheDocument();
```

- [ ] **Step 8: Check frontend type-checking**

Run: `cd web && npm run lint`
Expected: No type errors.

- [ ] **Step 9: Run frontend tests**

Run: `cd web && npm run test`
Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
git add web/src/
git commit -m "feat | web | remove ChatMode, rag_meta, updateConversation; handle retrieve tool_result (#228)"
```

---

### Task 8: Final cleanup — remove SSE_EVENT_RAG_META constant and DB migration safety

**Files:**
- Modify: `backend/src/bibilab/routers/chat.py`
- Check: all files for dead references

- [ ] **Step 1: Remove SSE_EVENT_RAG_META from backend**

Delete line 57 from `chat.py` (the `SSE_EVENT_RAG_META = "rag_meta"` constant). Already removed in the import list in Task 4 — verify.

- [ ] **Step 2: Full dead-reference sweep**

Run:
```bash
grep -rn "rag_meta\|SSE_EVENT_RAG_META\|classify_query\|params_for_type\|map_type_to_mode\|query_routing_enabled\|ChatMode\|CHAT_MODE\|QueryType\|QUERY_TYPE\|updateConversation\|PatchConversationRequest\|log_query_classification" backend/src/ web/src/ --include="*.py" --include="*.ts" --include="*.tsx" | grep -v __pycache__ | grep -v ".pyc" | grep -v node_modules | grep -v ".d.ts"
```

Expected: No hits (or only hits in test files that have been updated, and `search_mode` which is the new parameter name).

- [ ] **Step 3: Run full backend + frontend test suites**

```bash
uv run pytest -v --tb=short 2>&1 | tail -20
cd web && npm run lint && npm run test 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix | backend+web | final cleanup — remove SSE_EVENT_RAG_META, dead refs (#228)"
```

---

## Self-Review

### 1. Spec coverage

| Spec section | Covered by |
|---|---|
| RETRIEVE_TOOL definition | Task 2 Step 4 |
| search_mode_to_params | Task 2 Step 4 |
| execute_retrieve | Task 2 Step 4 |
| stream_with_tools loop | Task 3 Step 3 |
| chat.py event_generator rewrite | Task 4 Step 2 |
| System prompt rule 0 | Task 4 Step 1 |
| Delete route.py | Task 5 Step 1 |
| Delete QueryType/ChatMode/map_type_to_mode | Task 5 Step 2 |
| Delete query_routing_enabled | Task 5 Step 3 |
| Delete log_query_classification | Task 5 Step 4 |
| Drop query_classifications table | Task 5 Step 4 |
| Drop conversations.mode | Task 5 Step 4 |
| Delete PATCH endpoint | Task 5 Step 5 |
| Delete PatchConversationRequest | Task 5 Step 5 |
| Frontend ChatMode/rag_meta delete | Task 7 |
| Frontend tool_result → rag | Task 7 Step 5 |
| Frontend ObsChip search_mode | Task 7 Step 6 |
| Cleanup SSE_EVENT_RAG_META | Task 8 |

### 2. Placeholder scan

No TBDs, TODOs, or placeholder code. All steps have concrete code and commands.

### 3. Type consistency

- `RetrievalParams` stays in `_enums.py` with same fields (`depth_per_source`, `top_k`)
- `search_mode_to_params` returns `RetrievalParams` — matches old `params_for_type` signature with `str` key instead of `QueryType`
- `execute_tool` gains `cfg: BibilabConfig` parameter — all callers updated
- `ConverationResponse` loses `mode` field — frontend type matches
- `RagMetadata.mode` → `RagMetadata.search_mode` — ObsChip uses `rag.search_mode`
- `tool_result` event carries `name` + `result` — frontend checks `result.name === "retrieve"` to set rag
