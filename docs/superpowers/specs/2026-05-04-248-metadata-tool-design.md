# 248 — Metadata-Aggregation Retrieval Primitive

Date: 2026-05-04 | Status: spec

## Problem

Some user queries require structured metadata from the DB, not transcript search. From the #234 audit: "how many sources?", "longest video?", "language breakdown?" The `retrieve` tool searches chunk text and cannot answer these.

The #254 spike confirmed the LLM correctly distinguishes content questions from metadata questions — a metadata tool is needed.

## Design

### Tool: `query_list_metadata`

Same loopback pattern as `retrieve`. LLM calls it, gets pre-computed structured data, incorporates into answer. Operates on `source_ids` (the user's selected subset), same scope as `retrieve`.

```python
QUERY_LIST_METADATA_TOOL = ToolDefinition(
    name="query_list_metadata",
    description=(
        "Look up structured metadata about the sources you are chatting about. "
        "Use when the user asks about counts, durations, or languages. "
        "Do NOT use for content questions — use retrieve for those."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["count", "longest", "languages"],
                "description": (
                    "count = number of sources; "
                    "longest = source with the longest duration; "
                    "languages = count per language"
                ),
            },
        },
        "required": ["query_type"],
    },
)
```

### Per-mode returns

All computation is server-side (SQL). The LLM reads, not computes. All queries filter by `source_ids` — the same subset `retrieve` uses.

| query_type | Result shape |
|---|---|
| `count` | `{"count": 8}` |
| `longest` | `{"title": "...", "duration_seconds": 3600}` |
| `languages` | `{"languages": {"zh": 5, "en": 3, "unknown": 1}}` |

### Execution flow

```
User: "How many videos in this list?"
  → LLM calls query_list_metadata(query_type="count")
  → execute_query_list_metadata(source_ids, "count") → {"count": 8}
  → LLM: "There are 8 videos."
```

Loopback: yes. Results feed back into `stream_with_tools` for another LLM turn.

### Dispatch pattern

Thin per-mode helpers in `db.py` (query only, no logic). Dispatch in `chat_tools.execute_query_list_metadata`:

```python
# db.py — one function per mode, source_ids as parameter
async def count_sources(source_ids: list[str]) -> int: ...
async def longest_source(source_ids: list[str]) -> dict | None: ...
async def language_breakdown(source_ids: list[str]) -> dict[str, int]: ...

# chat_tools.py — dispatch
async def execute_query_list_metadata(source_ids: list[str], query_type: str) -> dict:
    if query_type == "count":
        count = await count_sources(source_ids)
        return {"count": count}
    if query_type == "longest":
        row = await longest_source(source_ids)
        return {"title": row["title"], "duration_seconds": row["duration_seconds"]} if row else {}
    if query_type == "languages":
        langs = await language_breakdown(source_ids)
        return {"languages": langs}
    logger.warning("Unknown query_type %r — falling back to count", query_type)
    return {"count": await count_sources(source_ids)}
```

This follows the project rule: `db.py` is strictly SQL queries; domain logic lives in pipeline/routers.

## Changes

| File | Change |
|---|---|
| `pipeline/chat_tools.py` | Add `QUERY_LIST_METADATA_TOOL`, `execute_query_list_metadata()`, wire into `execute_tool()` |
| `routers/chat.py` | Add to `ALL_TOOLS` and `LOOPBACK_TOOLS`; update `GROUNDING_SYSTEM_PROMPT` rule 0 to exclude counts from retrieve |
| `db.py` | Add `count_sources(source_ids)`, `longest_source(source_ids)`, `language_breakdown(source_ids)` |
| Tests | Per-mode correctness, empty source_ids, NULL language, scope (source_ids subset ≠ full list), dispatch fallback, LLM smoke test |

### System prompt diff

Rule 0 currently routes counts to `retrieve`:

```
- "summaries, counts across sources — you MUST call the retrieve tool BEFORE answering."
+ "summaries — you MUST call the retrieve tool BEFORE answering."
+ "For counts, durations, or language questions, call query_list_metadata instead."
```

## Edge cases

- **Empty source_ids** — `count` returns 0; `longest` returns `{}`; `languages` returns `{}`
- **Language is NULL** — `COALESCE(language, 'unknown')` in SQL
- **Multiple sources tie for longest** — returns first match
- **Scope correctness** — user selects 3 of 8 sources, asks "how many?" → tool returns 3

## Out of scope

- `shortest`, `durations`, `uploaders`, `total_duration`, `platforms` — no audit-backed demand; add when needed
- Per-source metadata beyond title + duration
- Frontend changes — metadata `tool_result` is ignored by the client (same pattern as `generate_report` tool_result). No RAG pill for metadata answers in v0; revisit if user confusion reported.
