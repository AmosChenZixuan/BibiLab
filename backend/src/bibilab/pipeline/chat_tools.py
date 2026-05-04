"""Tool definitions and execution for chat."""

import logging
import uuid

from bibilab.config import BibilabConfig
from bibilab.db import create_job
from bibilab.models._enums import RetrievalParams
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.embed import retrieve

logger = logging.getLogger(__name__)

_VALID_ARTIFACT_TYPES = frozenset({"brief", "study_guide", "blog_post", "custom_report"})

_PARAMS_BY_MODE = {
    "factual": RetrievalParams(depth_per_source=1, top_k=4),
    "breadth": RetrievalParams(depth_per_source=1, top_k=20),
    "analytical": RetrievalParams(depth_per_source=4, top_k=12),
}


def search_mode_to_params(search_mode: str, sources_total: int) -> RetrievalParams:
    if search_mode == "breadth" and sources_total < 3:
        return _PARAMS_BY_MODE["factual"]
    base = _PARAMS_BY_MODE.get(search_mode)
    if base is None:
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
        "source_coverage": [{"video_id": s.video_id, "title": s.video_title} for s in result.source_coverage],
        "_chunks": [
            {"title": c.video_title, "start": c.timestamp_start, "end": c.timestamp_end, "content": c.content}
            for c in result.chunks
        ],
    }


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
