"""Tool definitions and execution for chat."""

import json
import logging
import uuid
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import count_sources, create_job, language_breakdown, longest_source
from bibilab.models._enums import RetrievalParams
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.embed import apply_source_filter, retrieve

logger = logging.getLogger(__name__)


@dataclass
class CitationRegistryEntry:
    index: int
    source_id: str
    title: str = ""
    chunk_ids: set[str] = field(default_factory=set)


def _format_chunk_for_llm(chunk: dict, index: int) -> str:
    ts_start = int(chunk["start"])
    ts_end = int(chunk["end"])
    return f'[{index} @ {ts_start}s-{ts_end}s]: "{chunk["content"]}"'


def _build_source_headers(registry: dict[str, CitationRegistryEntry]) -> str:
    lines = []
    for entry in sorted(registry.values(), key=lambda e: e.index):
        title = entry.title
        lines.append(f'Source [{entry.index}]: "{title}"')
    return "\n".join(lines)


_VALID_ARTIFACT_TYPES = frozenset({"brief", "study_guide", "blog_post", "custom_report"})


DEFAULT_EXPECTED_HITS = "few"


def params_for_expected_hits(expected_hits: str) -> RetrievalParams:
    """Map expected_hits to RetrievalParams."""
    return {
        "one": RetrievalParams(depth_per_source=1, top_k=2),
        "few": RetrievalParams(depth_per_source=2, top_k=8),
        "many": RetrievalParams(depth_per_source=5, top_k=24),
    }[expected_hits]


def _format_filter_miss_message(filter_str: str, titles: list[str], *, cap: int = 20) -> str:
    shown = titles[:cap]
    overflow = len(titles) - cap
    overflow_suffix = f"\n- ... and {overflow} more" if overflow > 0 else ""
    titles_fmt = "\n".join(f"- {t}" for t in shown) + overflow_suffix
    return f"\n\nNo sources matched 'title_contains: {filter_str}'. Available titles:\n{titles_fmt}"


def _empty_retrieve_result(
    query: str,
    source_filter: dict | None,
    filter_miss: bool,
    sources_total: int,
    expected_hits: str = DEFAULT_EXPECTED_HITS,
    chunks_text: str = "",
) -> dict:
    return {
        "query": query,
        "source_filter": source_filter,
        "filter_miss": filter_miss,
        "expected_hits": expected_hits,
        "candidates_evaluated": 0,
        "sources_with_hits": 0,
        "sources_total": sources_total,
        "source_coverage": [],
        "_chunks": chunks_text,
        "_turn_indices": [],
        "_raw_chunks": [],
    }


GENERATE_REPORT_TOOL = ToolDefinition(
    name="generate_report",
    description=(
        "Generate a report/artifact from the user's selected sources. "
        "Use when the user asks to create a summary, study guide, blog post, or custom report."
    ),
    parameters={
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["brief", "study_guide", "blog_post", "custom_report"],
                "description": "The type of report to generate",
            },
            "prompt": {
                "type": "string",
                "description": "A description of what the report should cover",
            },
        },
        "required": ["type", "prompt"],
    },
)

RETRIEVE_TOOL = ToolDefinition(
    name="retrieve",
    description=(
        "Retrieve information from video transcripts. "
        "When the user names a specific source by title, episode number, "
        "or other identifier (e.g. '第八集', '第3道菜', 'the React video'), "
        "include source_filter so retrieval is scoped to that subset. "
        "Use expected_hits='one' for single-fact questions ('how many eggs'), "
        "'few' (default) for narrow content questions, "
        "'many' for survey questions or comprehensive summaries. "
        "Do NOT use for pure greetings (hi, thanks) or conversation-control messages."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — key terms or question in the user's language",
            },
            "source_filter": {
                "type": "object",
                "description": (
                    "REQUIRED when the user references a specific source by episode/chapter number, "
                    "title fragment, character/chef name, or video name. "
                    "Omit ONLY for cross-source queries (compare, list all, etc.). "
                    "Example: user says '第八集' → {'title_contains': '第八集'}"
                ),
                "properties": {
                    "title_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring match against sources.title",
                    },
                },
            },
            "expected_hits": {
                "type": "string",
                "enum": ["one", "few", "many"],
                "description": (
                    "Expected retrieval breadth. "
                    "one = single-fact, depth_per_source=1, top_k=2; "
                    "few = default, depth_per_source=2, top_k=8; "
                    "many = survey/summary, depth_per_source=5, top_k=24"
                ),
            },
        },
        "required": ["query"],
    },
)

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


async def execute_query_list_metadata(source_ids: list[str], query_type: str) -> dict:
    if query_type == "count":
        return {"count": await count_sources(source_ids)}
    if query_type == "longest":
        row = await longest_source(source_ids)
        return row if row is not None else {"title": None, "duration_seconds": None}
    if query_type == "languages":
        return {"languages": await language_breakdown(source_ids)}
    logger.warning("Unknown query_type %r — falling back to count", query_type)
    return {"count": await count_sources(source_ids)}


async def execute_generate_report(
    list_id: str,
    artifact_type: str,
    prompt: str,
    source_ids: list[str],
    ui_lang: str,
) -> dict:
    artifact_id = str(uuid.uuid4())
    job_id = await create_job(
        type="artifact",
        meta={
            "artifact_id": artifact_id,
            "list_id": list_id,
            "type": artifact_type,
            "prompt": prompt,
            "source_ids": source_ids,
            "ui_lang": ui_lang,
        },
    )
    return {
        "artifact_id": artifact_id,
        "job_id": job_id,
        "name": artifact_type,
        "type": artifact_type,
    }


async def execute_retrieve(
    query: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
    source_filter: dict | None = None,
    expected_hits: str = DEFAULT_EXPECTED_HITS,
) -> dict:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    # Some Anthropic-compatible providers (e.g. MiniMax) serialize nested object
    # arguments as JSON strings in later tool-call rounds. Attempt parse before use.
    if isinstance(source_filter, str):
        try:
            source_filter = json.loads(source_filter)
        except json.JSONDecodeError:
            logger.warning("source_filter was a non-JSON string: %r", source_filter)
            source_filter = None

    # Apply source_filter only when title_contains is present
    scoped_source_ids: list[str] | None = None
    if isinstance(source_filter, dict) and source_filter.get("title_contains"):
        filtered_source_ids, all_titles = await apply_source_filter(source_ids, source_filter)
        if not filtered_source_ids:
            filter_str = source_filter["title_contains"]
            msg = _format_filter_miss_message(filter_str, all_titles) if all_titles else ""
            return _empty_retrieve_result(
                query=query,
                source_filter=source_filter,
                filter_miss=True,
                sources_total=len(source_ids),
                expected_hits=expected_hits,
                chunks_text=msg,
            )
        scoped_source_ids = filtered_source_ids

    params = params_for_expected_hits(expected_hits)
    result = await retrieve(
        query_text=query,
        source_ids=source_ids,
        cfg=cfg,
        params=params,
        scoped_source_ids=scoped_source_ids,
    )

    # Assign indices: new sources get next available index
    next_index = max((e.index for e in registry.values()), default=0) + 1
    for s in result.source_coverage:
        sid = source_map.get(s.video_id)
        if sid is None:
            continue
        if sid not in registry:
            registry[sid] = CitationRegistryEntry(
                index=next_index,
                source_id=sid,
                title=s.video_title,
            )
            next_index += 1

    # Build video_id → registry index lookup for chunk formatting
    video_id_to_index: dict[str, int] = {}
    for s in result.source_coverage:
        sid = source_map.get(s.video_id)
        if sid and sid in registry:
            video_id_to_index[s.video_id] = registry[sid].index

    # Accumulate chunk_ids per source (synthetic key: video_id_start_end)
    for c in result.chunks:
        sid = source_map.get(c.video_id)
        if sid and sid in registry:
            cid = f"{c.video_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}"
            registry[sid].chunk_ids.add(cid)

    # Collect indices actually retrieved this turn (for the enumeration line)
    turn_indices = sorted(set(video_id_to_index.values()))

    chunks_formatted = []
    raw_chunks = []
    for c in result.chunks:
        if c.video_id not in video_id_to_index:
            continue
        idx = video_id_to_index[c.video_id]
        chunks_formatted.append(
            _format_chunk_for_llm(
                {"start": c.timestamp_start, "end": c.timestamp_end, "content": c.content},
                index=idx,
            )
        )
        raw_chunks.append(
            {
                "source_id": source_map.get(c.video_id, ""),
                "chunk_id": f"{c.video_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}",
                "content": c.content,
                "video_title": c.video_title,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
                "citation_index": idx,
            }
        )

    return {
        "query": query,
        "source_filter": source_filter,
        "filter_miss": False,
        "expected_hits": expected_hits,
        "candidates_evaluated": result.candidates_evaluated,
        "sources_with_hits": result.sources_with_hits,
        "sources_total": result.sources_total,
        "source_coverage": [
            {
                "source_id": source_map.get(s.video_id, ""),
                "video_id": s.video_id,
                "title": s.video_title,
            }
            for s in result.source_coverage
        ],
        "_chunks": (
            f"Sources retrieved this turn: {', '.join(f'[{i}]' for i in turn_indices)}. "
            "Cite only these indices.\n\n"
            f"{_build_source_headers(registry)}\n\n" + "\n".join(chunks_formatted)
        ),
        "_turn_indices": turn_indices,
        "_raw_chunks": raw_chunks,
    }


def build_tool_block_entry(
    tool_use_id: str,
    name: str,
    arguments: dict,
    result: dict,
    raw_chunks: list[dict] | None,
) -> dict:
    """Normalize an in-flight tool call+result into the persisted shape.

    For retrieve, internal underscore-prefixed fields are stripped (they're
    LLM-formatting metadata, not replay state); raw chunk snapshots are
    attached so replay survives re-embedding. For other tools, the result
    is stored as-is.
    """
    if name == RETRIEVE_TOOL.name:
        summary = {k: v for k, v in result.items() if not k.startswith("_")}
        return {
            "tool_use_id": tool_use_id,
            "name": name,
            "arguments": arguments,
            "result": {"chunks": raw_chunks or [], "summary": summary},
        }
    return {
        "tool_use_id": tool_use_id,
        "name": name,
        "arguments": arguments,
        "result": result,
    }


async def execute_tool(
    tool_name: str,
    arguments: dict,
    list_id: str,
    source_ids: list[str],
    ui_lang: str,
    cfg: BibilabConfig,
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
) -> dict:
    if tool_name == RETRIEVE_TOOL.name:
        logger.info(
            "retrieve tool call: query=%r source_filter=%r expected_hits=%r",
            arguments["query"],
            arguments.get("source_filter"),
            arguments.get("expected_hits", DEFAULT_EXPECTED_HITS),
        )
        return await execute_retrieve(
            query=arguments["query"],
            source_ids=source_ids,
            cfg=cfg,
            registry=registry,
            source_map=source_map,
            source_filter=arguments.get("source_filter"),
            expected_hits=arguments.get("expected_hits", DEFAULT_EXPECTED_HITS),
        )
    if tool_name == GENERATE_REPORT_TOOL.name:
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
    if tool_name == QUERY_LIST_METADATA_TOOL.name:
        return await execute_query_list_metadata(
            source_ids=source_ids,
            query_type=arguments["query_type"],
        )
    raise ValueError(f"Unknown tool: {tool_name}")


def expand_message_for_provider(
    msg: dict,
    protocol: str,  # "anthropic" or "openai"
) -> list[dict]:
    """Expand a stored message into provider-shape messages.

    For text-only messages (no tool_blocks, or empty), returns [msg] without
    the tool_blocks key. For assistant messages with tool_blocks, returns the
    synthetic shape the LLM expects so it sees prior tool_use/tool_result
    blocks on subsequent turns.
    """
    blocks = msg.get("tool_blocks")
    if not blocks:
        # Strip tool_blocks if present-but-empty; producer expects clean shape.
        clean = {k: v for k, v in msg.items() if k != "tool_blocks"}
        return [clean]

    text = msg.get("content", "")

    if protocol == "anthropic":
        assistant_content: list[dict] = []
        tool_result_content: list[dict] = []
        for b in blocks:
            tool_use_id = b.get("tool_use_id")
            name = b.get("name")
            arguments = b.get("arguments")
            result = b.get("result")
            if tool_use_id is None or name is None or arguments is None or result is None:
                logger.warning("expand_message_for_provider skipping malformed block: missing keys")
                continue
            assistant_content.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": arguments})
            tool_result_content.append(
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": json.dumps(result)}
            )
        if text:
            assistant_content.append({"type": "text", "text": text})

        return [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_result_content},
        ]

    if protocol == "openai":
        openai_tool_calls: list[dict] = []
        out: list[dict] = [
            {"role": "assistant", "content": text or None, "tool_calls": openai_tool_calls},
        ]
        for b in blocks:
            tool_use_id = b.get("tool_use_id")
            name = b.get("name")
            arguments = b.get("arguments")
            result = b.get("result")
            if tool_use_id is None or name is None or arguments is None or result is None:
                logger.warning("expand_message_for_provider skipping malformed block: missing keys")
                continue
            openai_tool_calls.append(
                {
                    "id": tool_use_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": json.dumps(result),
                }
            )
        return out

    logger.warning("expand_message_for_provider unknown protocol=%s — returning text-only fallback", protocol)
    clean = {k: v for k, v in msg.items() if k != "tool_blocks"}
    return [clean]


def reseed_citation_registry(
    registry: dict[str, CitationRegistryEntry],
    history: list[dict],
) -> None:
    """Reseed the citation registry from stored retrieve tool_blocks in history.

    Walks each assistant message's tool_blocks. For each retrieve result,
    re-creates CitationRegistryEntry instances keyed by source_id so prior
    [N] markers in old assistant text continue to resolve to the same source.
    """
    for msg in history:
        for block in msg.get("tool_blocks") or []:
            if block.get("name") != RETRIEVE_TOOL.name:
                continue
            chunks = block.get("result", {}).get("chunks", [])
            for ch in chunks:
                sid = ch.get("source_id")
                if not sid:
                    continue
                ci = ch.get("citation_index")
                if ci is None:
                    continue
                entry = registry.get(sid)
                if entry is None:
                    entry = CitationRegistryEntry(
                        index=ci,
                        source_id=sid,
                        title=ch.get("video_title", ""),
                    )
                    registry[sid] = entry
                cid = ch.get("chunk_id")
                if cid:
                    entry.chunk_ids.add(cid)
