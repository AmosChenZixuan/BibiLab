"""Tool definitions and execution for chat."""

import json
import logging
import uuid
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import count_sources, create_job, get_titles, language_breakdown, longest_source
from bibilab.models._enums import RetrievalParams
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.embed import retrieve

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
        "When the user names a specific source by episode number, title, or other identifier "
        "(e.g. '第八集', '第3道菜', 'the React video'), "
        "first call query_list_metadata(query_type='titles') to get the list of sources, "
        "then call retrieve with source_ids set to the matching source ID(s). "
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
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
                "description": (
                    "Optional list of source IDs to search within. "
                    "When the user references a specific source(s) by episode #, title, chef/character name, etc., "
                    "first call query_list_metadata(query_type='titles') to get the list of sources, "
                    "then pass the selected source_ids here. "
                    "Omit for cross-source queries (compare, list all, etc.). "
                    "Example: query_list_metadata returns [{source_id: 's8', title: '第8集 xxx'}, ...]; "
                    "if the user asks about '第八集', call retrieve(source_ids=['s8']). "
                ),
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
                "enum": ["count", "longest", "languages", "titles"],
                "description": (
                    "count = number of sources; "
                    "longest = source with the longest duration; "
                    "languages = count per language; "
                    "titles = list of {source_id, title} for source selection"
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
    if query_type == "titles":
        return {"titles": await get_titles(source_ids)}
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
    selected_source_ids: list[str] | None = None,
    expected_hits: str = DEFAULT_EXPECTED_HITS,
) -> dict:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    # selected_source_ids: list of strings selected by LLM via query_list_metadata.
    # Compute intersection with available source_ids to produce scoped_source_ids.
    scoped_source_ids: list[str] | None = None
    if selected_source_ids:  # non-None and non-empty; [] means "search all"
        id_set = set(source_ids)
        scoped_source_ids = [sid for sid in selected_source_ids if sid in id_set]
        if len(scoped_source_ids) < len(selected_source_ids):
            missing = set(selected_source_ids) - id_set
            logger.debug("Some selected_source_ids not in source_ids pool, filtered out: %s", missing)

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
        source_ids_arg = arguments.get("source_ids")
        # Validate source_ids type: must be None or list of str
        if source_ids_arg is not None and not isinstance(source_ids_arg, list):
            logger.warning("retrieve source_ids=%r is not a list, ignoring", source_ids_arg)
            source_ids_arg = None
        logger.info(
            "retrieve tool call: query=%r source_ids=%r expected_hits=%r",
            arguments["query"],
            source_ids_arg,
            arguments.get("expected_hits", DEFAULT_EXPECTED_HITS),
        )
        return await execute_retrieve(
            query=arguments["query"],
            source_ids=source_ids,
            cfg=cfg,
            registry=registry,
            source_map=source_map,
            selected_source_ids=source_ids_arg,
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
