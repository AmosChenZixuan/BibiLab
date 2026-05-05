# 248 — Metadata-Aggregation Retrieval Primitive

Date: 2026-05-04 | Status: spec

## Problem

Some user queries require structured metadata from the DB, not transcript search. From the #234 audit: "how many sources?", "longest video?", "language breakdown?" The `retrieve` tool searches chunk text and cannot answer these.

The #254 spike confirmed the LLM correctly distinguishes content questions from metadata questions and falls back to explaining the limitation — a metadata tool is needed.

## Design

### Tool: `query_list_metadata`

Same loopback pattern as `retrieve`. LLM calls it, gets pre-computed structured data, incorporates into answer.

```python
QUERY_LIST_METADATA_TOOL = ToolDefinition(
    name="query_list_metadata",
    description=(
        "Look up structured metadata about the sources in the current list. "
        "Use when the user asks about counts, durations, languages, or uploaders. "
        "Do NOT use for content questions — use retrieve for those."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["count", "longest", "shortest", "durations", "languages", "uploaders"],
                "description": (
                    "count = total number of sources; "
                    "longest = single source with max duration; "
                    "shortest = single source with min duration; "
                    "durations = all titles with durations (for comparison); "
                    "languages = count per language; "
                    "uploaders = count per uploader"
                ),
            },
        },
        "required": ["query_type"],
    },
)
```

### Per-mode returns

All computation is server-side (SQL). The LLM reads, not computes.

| query_type | SQL | Result shape |
|---|---|---|
| `count` | `SELECT COUNT(*) FROM sources WHERE list_id=?` | `{"count": 8}` |
| `longest` | `SELECT title, MAX(duration_seconds) ...` | `{"title": "...", "duration_seconds": 3600}` |
| `shortest` | `SELECT title, MIN(duration_seconds) ...` | `{"title": "...", "duration_seconds": 120}` |
| `durations` | `SELECT title, duration_seconds ... ORDER BY duration_seconds DESC` | `{"sources": [{"title": "...", "duration_seconds": N}, ...]}` |
| `languages` | `SELECT language, COUNT(*) ... GROUP BY language` | `{"languages": {"zh": 5, "en": 3}}` |
| `uploaders` | `SELECT uploader, COUNT(*) ... GROUP BY uploader` | `{"uploaders": {"NameA": 3, "NameB": 5}}` |

### Execution flow

```
User: "How many videos in this list?"
  → LLM calls query_list_metadata(query_type="count")
  → execute_query_list_metadata(list_id, "count") → {"count": 8}
  → LLM: "There are 8 videos in this list."
```

Loopback: yes. Same as `retrieve` — results feed back into `stream_with_tools` for another LLM turn.

## Changes

| File | Change |
|---|---|
| `pipeline/chat_tools.py` | Add `QUERY_LIST_METADATA_TOOL`, `execute_query_list_metadata()`, wire into `execute_tool()` |
| `routers/chat.py` | Add to `ALL_TOOLS` list and `LOOPBACK_TOOLS` set |
| `db.py` | Add `get_list_metadata(list_id, query_type)` |
| Tests | Test each mode, test dispatch fallback, test smoke integration |

## Edge cases

- **Empty list** — `count` returns 0; `longest`/`shortest`/`durations` return `null`; `languages`/`uploaders` return `{}`
- **Unknown query_type** — fall back to `count` with warning log (matches `search_mode_to_params` pattern)
- **Language is NULL** — grouped as `"unknown"`
- **Multiple sources tie for longest/shortest** — returns first match

## Out of scope

- `total_duration`, `platforms`, `by_uploader` list, or other aggregate modes — add when user demand appears
- Per-source metadata beyond title + duration
- Frontend changes — metadata `tool_result` is ignored by the client (same as current `generate_report` tool_result)
