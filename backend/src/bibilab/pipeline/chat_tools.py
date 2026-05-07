"""Tool definitions and execution for chat."""

import logging
import uuid
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import count_sources, create_job, language_breakdown, longest_source
from bibilab.models._enums import RetrievalParams, SearchMode
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

_PARAMS_BY_MODE = {
    "factual": RetrievalParams(depth_per_source=1, top_k=4),
    "breadth": RetrievalParams(depth_per_source=1, top_k=20),
    "analytical": RetrievalParams(depth_per_source=4, top_k=12),
}


def search_mode_to_params(search_mode: SearchMode, sources_total: int) -> RetrievalParams:
    if search_mode == "breadth" and sources_total < 3:
        return _PARAMS_BY_MODE["factual"]
    base = _PARAMS_BY_MODE.get(search_mode)
    if base is None:
        logger.warning("Unknown search_mode %r — falling back to factual", search_mode)
        return _PARAMS_BY_MODE["factual"]
    if search_mode == "breadth":
        return RetrievalParams(depth_per_source=base.depth_per_source, top_k=min(base.top_k, sources_total))
    if search_mode == "factual":
        return base
    top_k = max(base.top_k, min(sources_total, base.top_k * 3))
    return RetrievalParams(depth_per_source=base.depth_per_source, top_k=top_k)


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
    search_mode: SearchMode,
    source_ids: list[str],
    cfg: BibilabConfig,
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
) -> dict:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    params = search_mode_to_params(search_mode, len(source_ids))
    result = await retrieve(query_text=query, source_ids=source_ids, cfg=cfg, params=params)

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

    chunks_formatted = [
        _format_chunk_for_llm(
            {"start": c.timestamp_start, "end": c.timestamp_end, "content": c.content},
            index=video_id_to_index[c.video_id],
        )
        for c in result.chunks
        if c.video_id in video_id_to_index
    ]

    return {
        "query": query,
        "search_mode": search_mode,
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
    if tool_name == "retrieve":
        return await execute_retrieve(
            query=arguments["query"],
            search_mode=arguments["search_mode"],
            source_ids=source_ids,
            cfg=cfg,
            registry=registry,
            source_map=source_map,
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
    if tool_name == "query_list_metadata":
        return await execute_query_list_metadata(
            source_ids=source_ids,
            query_type=arguments["query_type"],
        )
    raise ValueError(f"Unknown tool: {tool_name}")
