# Chat & Retrieval Architecture

How a chat turn works end to end: request lifecycle, retrieval, tool loop, streaming, persistence, compression, and the eval/observability surfaces. The rules an AI must follow when *changing* this code live in `backend/CLAUDE.md`; this document explains how the system behaves.

## Request lifecycle

```
POST /lists/:id/chat (SSE) — creates user(pending) + assistant(streaming) rows atomically, spawns async producer
  → asyncio.Task: run_chat_turn writes events into StreamBuffer
  → POST handler returns SSE stream consuming from buffer (late-subscriber-safe replay)
  → producer's finally: update_turn_terminal flips BOTH rows to the same terminal status
    AND clears active_stream_message_id in one transaction (fallback set_active_stream
    clear if that transaction fails, else the conversation wedges at 409)
  → producer fires fire-and-forget: evict buffer after grace, compress if done

GET  /lists/:id/chat/{msg_id}/stream — reattach to an active stream (204 if evicted)
POST /lists/:id/chat/{msg_id}/cancel — cancel producer task, flips turn to status='cancelled'

Startup: sweep_orphaned_streams flips leftover in-flight rows (pending + streaming) to failed
Shutdown: cancel all active tasks, drain with 5s timeout
```

- **Producer/consumer split**: `run_chat_turn` (async Task) writes SSE events into `StreamBuffer`; `_sse_consumer` reads from the buffer. Decouples LLM lifetime from HTTP request lifetime.
- **Lifecycle sweeps**: startup flips leftover in-flight rows (`'streaming'` + `'pending'`, via `IN_FLIGHT_MESSAGE_STATUSES`) to `failed`. Shutdown cancels all tasks and drains with a 5s timeout. Buffer eviction fires 60s after terminal status.

## SSE events

`meta`, `delta`, `citation`, `tool_call_start`, `tool_result`, `rag`, `done`, `error`, `cancelled`

`meta` is the first event of every stream, carrying `{message_id}` so the client can wire cancel-by-id before the first delta (constant `SSE_EVENT_META`). The other eight are also `stream_llm` discriminators; `meta` is emitted only by the HTTP handler.

- Preamble text generated before a loopback tool call is streamed to the client immediately (parsed incrementally via `parse_delta`), not buffered until stream end. Trade-off: short filler like "Let me look that up…" reaches the client before the retrieve runs.
- `[N]` citations are streamed as `citation` SSE events: `parse_delta` strips the `[N]` markers from LLM deltas and emits `{index, section_id, source_id, timestamp_start, chunk_ids}` interleaved with `delta` events.
- Error events carry a machine-readable code in the `message` field (e.g. `llm_rate_limit_error`) for frontend i18n, mapped by `_classify_llm_error()` from SDK exceptions.
- **Final `rag` event**: in the `finally` block, after `context[]` is reconstructed from the citation registry, `run_chat_turn` emits one `rag` SSE event carrying the authoritative persisted-shape `rag.calls` just before the terminal event. The client replaces its incrementally-built ledger so expand works post-stream without a manual reload (the streaming `tool_result` payload omits `context[]`).

## System prompt

- **Grounding prompt**: `build_grounding_prompt(response_language)` produces a four-section markdown document: `## Workflow` (tool routing), `## Grounding` (excerpts-only), `## Citation` (`[N]` markers), `## Style` (direct, no follow-up). `response_language` is mapped to a native display name and interpolated once, as the tail directive (`Respond in X.`) governing all output including refusals. Teaches the LLM subject decomposition — one subject → one tool, multiple subjects (comparisons, multi-episode) → parallel calls, one per subject, picking the appropriate tool per subject.
- **Source scoping**: no per-turn source list in the prompt. The system prompt is static-per-language, so provider prefix caching covers the whole grounding prefix. Search scope is set solely by deterministic facet matching: the LLM passes `sequence_number` / `season_number` to `find_passages` / `read_section` only when the question references them; the backend matches them against `sources`. When facet matching finds no source (fail-open to the full pool), the tool result carries `_NO_MATCH_NOTE` as a fact-only LLM-visible prefix so the model states the degraded scope before answering.
- Fact/directive split: tools emit FACTS, the prompt owns DIRECTIVES.

## Retrieval

Two-tool surface:

- `find_passages` — recall-biased locator, returns section-keyed excerpts.
- `read_section` — ranking-immune bounded verbatim read of one section by `[N]`.

Pipeline per `find_passages` call:

1. **Deterministic facet scoping** is the sole pre-retrieval narrowing: `sequence_number`/`season_number` narrow the full source pool before search; fail-open on zero-match or facet-DB-error → full pool, `facet_scope.no_match` + LLM-visible note. The LLM *extracts* facets; the backend *matches* them deterministically. `series_name` matching is deferred.
2. **Hybrid search**: BM25 + vector, fused via RRF (k=60); dynamic pool = min(max(sources_total×3, params.top_k, 10), 60).
3. **Cross-encoder rerank**: Xenova/bge-reranker-base int8 (XLM-RoBERTa, zh+en), single spec `RERANKER_SPEC_ID = "bge-reranker-base-q"` (~4× smaller, ~1.85× faster on CPU than fp32, which was dropped). Its ONNX session sources providers from `interpreting_providers()` (shared with embed), which excludes compiler-based EPs — on macOS that drops CoreML, whose per-input-shape JIT recompile otherwise hangs >90s / OOM-kills the reranker on first chat retrieve.
4. **Gateless top-k by rerank order** (`FIND_PASSAGES_TOP_K = 8`): no relevance gate, no diversity cap, no neighbor-pull — rerank is ordering, not authority.
5. **Section-keyed fence render with speaker-turn reconstruction**: for the retained top-k only, `execute_find_passages` runs one batched `get_segments_for_ranges` query over each chunk's `(source_id, seg_start..seg_end)` and renders the LLM-facing fence body + `context[].preview` as grouped speaker turns via the shared `format_turns` helper (`[S{N}·SPK{k} @MM:SS]`). Speakers are namespaced per source by citation index `N`. Rerank + chroma/FTS still use raw `chunk.content` (embedder parity) — reconstruction is presentation-only.

Retrieve result `_chunks` are grouped per section under `===== [N] "title" · Section M =====` fences, then formatted as citation-ready `[N]: "content"` strings for the LLM; they are stripped from the client-bound `tool_result` SSE payload by `strip_internal()`. Within a fence, fragments render in chronological (segment) order, not rerank order, with a `[…]` gap marker between non-seg-adjacent fragments to mark elided transcript.

There is no source-list / `exclude_source_ids` index decision (static-per-language prompt, cacheable); `scoped_pool_size` is the full pool — see `facet_scope.matched_count` for the facet-narrowed count.

## Tool-calling loop

```
stream_with_tools(stream_llm loop):
    LLM yields tool_call (find_passages) → execute_find_passages() → hybrid_search (BM25 + vector RRF) → rerank → top-k (no gate) → section-keyed fence render
    → tool_call_start emitted → tool_result (with citation-formatted _chunks) fed back to LLM → second turn yields text with [N] citations
    → parse_delta strips [N] markers, emits citation SSE events ({index, section_id, source_id, timestamp_start, chunk_ids}) interleaved with delta events
    LLM yields tool_call (read_section) → execute_read_section() → resolves to one section by [N] → bounded verbatim transcript
    → tool_result fed back to LLM
    LLM yields delta + done directly (no tool) → exit loop
→ yield delta + tool_call_start + tool_result + citation + done events (SSE)
→ persist assistant message with rag metadata → asyncio.create_task: maybe_compress_conversation
```

- `stream_with_tools` wraps `stream_llm` in a bounded loop (max 3 iterations). This eliminated an 8K-token pre-stream classifier; the LLM decides which retrieval tool (if any) to use based on the question. Both tools are loopback: their results feed back for another LLM turn (allowing `find_passages → read_section` escalation).
- Any retrieve-family tool call disables all three for the rest of the turn.
- **Exhausted iterations** stop *executing* tools but keep them advertised, and inject a synthesis directive to force a prose answer instead of a hard error. Keeping tools advertised keeps the serving layer's tool-call grammar active, so a stubborn tool attempt parses as an ignored structured tool_call instead of leaking native tool-call tokens as the user-visible answer; execution is gated by `is_synthesis_turn`, not by withholding tools.
- **Empty synthesis**: if tools were used but no text was generated, one forced follow-up LLM call (tools still advertised, not executed; stray tool_calls dropped) tries to produce an answer. If the turn *still* yields no visible text it raises a typed no-text error (never persists a blank message) — `LLMOutputBudgetExceededError` → `llm_output_budget_exceeded` when the terminal stop_reason is a length cutoff, else `LLMEmptyResponseError` → `llm_empty_response`.
- **Subject decomposition + parallel retrieval**: the grounding prompt instructs the LLM to first decompose the user's message into distinct subjects (entities, episodes, items compared). One subject → one retrieval call (no hedging). Multiple subjects (`'A 和 B 的区别'`, `'第一集 xxx 第三集 yyy'`) → parallel calls, one per subject, picking the appropriate tool per subject. The `parallel_retrieve` log records the hedge-vs-decomposed rate when the retrieve family fires >1 call in one iteration, for offline analysis.
- **Cross-turn reuse**: prior-turn tool exchanges (find_passages, read_section) are dropped entirely from history replay by `expand_message_for_provider`; only synthesized prose survives into the next turn's LLM context. To answer from prior evidence the LLM must retrieve again this turn. The current turn's fresh tool calls are full-text (they go through `stream_with_tools`, not `expand_message_for_provider`).
- `stream_llm` supports both OpenAI and Anthropic protocols via `AsyncOpenAI`/`AsyncAnthropic`.

## Persisted rag ledger (`messages.metadata`)

JSON blob, nullable. Shape:

```
{"content_blocks": [...],
 "rag": {"calls": [{"tool_name", "query" | null, "section_id"?, "source_id"?, "source_title"?,
                    "section_coverage" (find_passages only, per section),
                    "context" (find_passages only, reconstructed at terminal rag event from citation registry),
                    "candidates_evaluated", "sources_with_hits", "sources_total", "reranked",
                    "scoped_pool_size", "facet_scope"}]}}
```

`tool_name` is `"find_passages"` or `"read_section"`; `read_section` rows have `query: null`, empty `context[]`, and `section_id`/`source_id`/`source_title` set. `scoped_pool_size` is the **full source pool**; `facet_scope.matched_count` carries the facet-narrowed count. Set by `run_chat_turn` post-stream from tool `tool_result` events. No migration for legacy persisted messages — the ledger renders best-effort from whatever fields exist.

## Conversation compression

Triggered when message count > 30; keeps a sliding window of 10; summarizes older messages via `_call_llm` in `asyncio.create_task`; old messages are deleted after summarization. The summary is prose only — the compression prompt does **not** preserve `[N]` markers (deliberate; see `docs/citation_system.md`). Only post-window messages retain live citations; the summary is injected into the system prompt on subsequent requests.

## Eval endpoint (`POST /eval/run_chat`)

Stateless one-shot JSON chat for eval frameworks. The engine is shared imported code (`build_grounding_prompt`, `stream_with_tools`, `execute_tool`, model gate, error classifier) — pipeline changes reach it automatically. Input differs by design: no history/summary replay (single bare message — prod's multi-turn assembly is never exercised here), `language` = request value or `"en"` (ignores `cfg.ai.output_language`), optional `llm` override field-merged onto `cfg.ai` (422 detail stays messages-only; `str(e)` would leak `api_key` via the merged dump). Output differs by design: full retrieved set per tool call with `cited` flags + `full_text` evidence and a response-level `llm_context` (the exact LLM-bound tool message per call, captured pre-strip — what a grader judges against), vs prod's cited-only + `preview`; rows snapshot at `tool_result` time since registry `full_text` is last-writer-wins across a turn. Mirrored literals to keep in sync when touching prod: the `"\n\n"` break around tool calls and the `"tool_error"` code. `CitationRegistryEntry.full_text` = post-dedup grounding text the LLM saw for a section; in-memory only, never persisted or SPA-visible.

`POST /eval/llm` is a bare `_call_llm` passthrough with the same llm-override merge, so the eval package needs no LLM SDK.

## Prompt-trace observability (opt-in)

`rag.debug_prompts: true` writes one JSON per chat turn at `~/.bibilab/debug/{message_id}.json` capturing the final cumulative LLM state (system, tools, messages, response, model, timestamp). The chat frontend shows a `</>` icon on assistant bubbles when both `debug_prompts` and the per-message `has_dump` flag are true; clicking opens a right-side drawer with envelope-aware rendering (Styled/Raw toggle). Storage write site: end of `run_chat_turn`. Best-effort: write errors logged as `dump_turn_failed`, never propagated.
