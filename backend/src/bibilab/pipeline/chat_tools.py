"""Tool definitions and execution for chat."""

import json
import logging
import uuid
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import count_sources, create_job, get_source_facets, language_breakdown, longest_source
from bibilab.models._enums import RetrievalParams
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.digest import parse_facet_int
from bibilab.pipeline.embed import retrieve

logger = logging.getLogger(__name__)

# Prepended to the LLM-visible retrieve result when facet scoping found no
# matching source and failed open to the full pool. The deleted source list
# (#310) was the LLM's only "episode not in library" signal; this restores it
# LLM-side. English by design: _chunks is LLM-facing and already English;
# build_grounding_prompt drives the user-visible response language separately.
_NO_MATCH_NOTE = (
    "No source matched the requested episode/season; searched all sources instead — say so before answering."
)

# Prepended to the LLM-visible retrieve result when retrieve returned zero
# chunks above the relevance gate. Anchors refusal to evidence: the LLM only
# learns the library lacks coverage from a completed retrieve, never from
# pre-retrieve judgment. English by design — see _NO_MATCH_NOTE.
_NO_COVERAGE_NOTE = (
    "Retrieve returned zero excerpts relevant to this query. The library does "
    "not cover this topic. Tell the user in their response language that the "
    "library has no content on this topic, and stop. Do not provide outside "
    "knowledge, real-world analogies, or encyclopedic definitions."
)


@dataclass
class CitationRegistryEntry:
    index: int
    source_id: str
    title: str = ""
    chunk_ids: set[str] = field(default_factory=set)
    # Populated at SSE-build time from execute_retrieve chunk data.
    # Used to reconstruct context[] for persisted metadata.
    first_chunk_id: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    rerank_score: float | None = None
    preview: str | None = None


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


def _build_fenced_chunks(
    chunks_by_index: dict[int, list[str]],
    registry: dict[str, CitationRegistryEntry],
) -> str:
    """Render chunks grouped by citation index, each group fenced by its source.

    Buckets are emitted in ascending index order; the caller-supplied order
    within each bucket (rerank order) is preserved. The fence makes the
    source boundary structural so the LLM does not graft a proper noun from
    one source onto another (#297).
    """
    title_by_index = {e.index: e.title for e in registry.values()}
    blocks = []
    for idx in sorted(chunks_by_index):
        title = title_by_index.get(idx, "")
        header = f'===== Source [{idx}]: "{title}" ====='
        blocks.append(header + "\n" + "\n".join(chunks_by_index[idx]))
    return "\n\n".join(blocks)


_VALID_ARTIFACT_TYPES = frozenset({"brief", "study_guide", "blog_post", "custom_report"})


_TOOL_NAME_TO_PARAMS: dict[str, RetrievalParams] = {
    "retrieve": RetrievalParams(depth_per_source=2, top_k=8, mode="narrow"),
    "survey": RetrievalParams(depth_per_source=5, top_k=24, mode="survey"),
    "retrieve_scoped": RetrievalParams(depth_per_source=2, top_k=8, mode="narrow"),
}

RETRIEVE_TOOL_NAMES: frozenset[str] = frozenset(_TOOL_NAME_TO_PARAMS.keys())


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
        "Retrieve from video transcripts for a single-fact / narrow content question.\n\n"
        "Extract keywords verbatim from the user's message. Copy proper nouns and "
        "technical terms exactly.\n\n"
        "Use this tool when the user asks a specific question with a clear answer — "
        "definitions, dates, names, single events, 'what is X', 'when did X happen', "
        "'who is X'.\n\n"
        "If the user explicitly references an episode or season (第八集, episode 3, "
        "第二季), use retrieve_scoped instead.\n\n"
        "If the user asks a survey / list / comparison question, use survey instead.\n\n"
        "Examples:\n"
        '  retrieve(query="拉格朗日点 稳定性 证明")\n'
        '  retrieve(query="长期情景记忆")'
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords extracted verbatim from the user's question",
            },
        },
        "required": ["query"],
    },
)

SURVEY_TOOL = ToolDefinition(
    name="survey",
    description=(
        "Retrieve from video transcripts for a broad / list-summary / comparison "
        "question.\n\n"
        "Broad questions use umbrella terms ('面食', 'political philosophy') that "
        "rarely appear verbatim in transcripts — the source says '牛肉面', '米粉'. "
        "Expand the query into subtypes and synonyms of the terms IN THE CURRENT "
        "USER MESSAGE, using general knowledge of the topic. The retrieval pool is "
        "larger (top_k=24, depth_per_source=5) than retrieve.\n\n"
        "Expand ONLY from the current user message. Do NOT borrow specific names, "
        "dishes, characters, or entities mentioned in earlier conversation turns — "
        "those belong to a different question and pollute this retrieval.\n\n"
        "Use this tool when the user asks for an overview, a list of items, a "
        "summary, a comparison, or anything that expects multiple sources — "
        "'what are the ways to X', 'compare A and B', 'list all the X', 'summarize'.\n\n"
        "If the user explicitly references an episode or season, use retrieve_scoped "
        "instead.\n\n"
        "Examples:\n"
        "  # user asks '有哪些面食做法' — expand the umbrella term '面食'\n"
        '  survey(query="面食 面条 牛肉面 拉面 米粉 馒头 饺子 做法")\n'
        "  # user asks '介绍政治哲学的流派' — expand '政治哲学'\n"
        '  survey(query="政治哲学 多元主义 自由主义 民主 思想流派")'
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Subtype / synonym expansion of the CURRENT user message's terms. "
                    "Do not borrow entities from prior turns."
                ),
            },
        },
        "required": ["query"],
    },
)

RETRIEVE_SCOPED_TOOL = ToolDefinition(
    name="retrieve_scoped",
    description=(
        "Retrieve from video transcripts, scoped to a specific episode or season.\n\n"
        "Use this tool ONLY when the current user message explicitly references an "
        "episode (第八集, episode 3, part 5) or a season (第二季, season 2). "
        "Do not infer the scope from prior conversation turns — if the current "
        "message has no explicit reference, use retrieve instead.\n\n"
        "Pass sequence_number for episode references and season_number for season "
        "references. Pass both when the user references both (第二季第八集).\n\n"
        "Examples:\n"
        '  retrieve_scoped(query="女巫 死期", sequence_number=5)\n'
        '  retrieve_scoped(query="第二季总览", season_number=2)\n'
        '  retrieve_scoped(query="主要事件", sequence_number=3, season_number=2)'
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords extracted from the user's question",
            },
            "sequence_number": {
                "type": "integer",
                "description": "Episode / part number — only if explicitly named in the CURRENT user message",
            },
            "season_number": {
                "type": "integer",
                "description": "Season number — only if explicitly named in the CURRENT user message",
            },
        },
        "required": ["query"],
    },
)

QUERY_LIST_METADATA_TOOL = ToolDefinition(
    name="query_list_metadata",
    description=(
        "Look up structured metadata about the sources. "
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


def _facet_int(v: object, key: str) -> int | None:
    """Coerce an LLM facet arg via the shared `parse_facet_int` primitive,
    degrading unusable values to None (a bad LLM guess drops the predicate,
    never raises — same best-effort contract as the digest path).

    Single-sources the coercion rules (>=1, bool/non-integral rejected) in
    parse_facet_int; only the degrade-and-log wrapper lives here.
    """
    try:
        return parse_facet_int(v)
    except ValueError:
        logger.warning("retrieve: %s=%r unusable, dropping predicate", key, v)
        return None


async def execute_retrieve(
    query: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    tool_name: str,
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
) -> dict:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    params = _TOOL_NAME_TO_PARAMS[tool_name]
    pool_size = len(source_ids)

    # Deterministic facet scoping (#309). Facet matching is the sole
    # pre-retrieval narrowing (#310 removed exclude/whitelist). Fail-open:
    # zero match (or facet-subquery DB error) → full pool, never empty.
    # scoped_pool_size is the full pool by design — see facet_scope.matched_count.
    facet_predicates = {
        k: v for k, v in (("sequence_number", sequence_number), ("season_number", season_number)) if v is not None
    }
    facet_matched_count: int | None = None
    facet_no_match = False
    scoped_source_ids: list[str] | None = None
    if facet_predicates:
        try:
            facets = await get_source_facets(source_ids)
        except Exception:  # noqa: BLE001 - facet scoping is best-effort; a DB
            # error here must fail open to the full pool, not nuke an
            # otherwise-fine retrieve (consistent with the zero-match contract).
            logger.warning("retrieve: get_source_facets failed, fail-open to full pool", exc_info=True)
            facets = {}
        matched = [
            sid
            for sid in source_ids
            if sid in facets and all(facets[sid].get(k) == v for k, v in facet_predicates.items())
        ]
        facet_matched_count = len(matched)
        if matched:
            scoped_source_ids = matched
        else:
            facet_no_match = True
            logger.warning(
                "retrieve: facet %s matched 0 sources, fail-open to full pool",
                facet_predicates,
            )

    logger.info(
        "retrieve dispatch: query=%r params(top_k=%d depth=%d) pool_size=%d",
        query,
        params.top_k,
        params.depth_per_source,
        pool_size,
    )
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

    # Accumulate chunk_ids per source (synthetic key: video_id_start_end).
    # Entry-level fields (preview / timestamps / rerank_score) are seeded from
    # the first HIT per source — never from a neighbor — so the persisted
    # context[] describes the gated chunk, not its surrounding context.
    for c in result.chunks:
        sid = source_map.get(c.video_id)
        if sid and sid in registry:
            cid = f"{c.video_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}"
            registry[sid].chunk_ids.add(cid)
            entry = registry[sid]
            if not getattr(c, "is_neighbor", False) and entry.timestamp_start is None:
                entry.first_chunk_id = cid
                entry.timestamp_start = c.timestamp_start
                entry.timestamp_end = c.timestamp_end
                entry.rerank_score = c.score
                entry.preview = c.content

    # Collect indices actually retrieved this turn (for the enumeration line)
    turn_indices = sorted(set(video_id_to_index.values()))

    chunks_by_index: dict[int, list[str]] = {}
    raw_chunks = []
    for c in result.chunks:
        if c.video_id not in video_id_to_index:
            continue
        idx = video_id_to_index[c.video_id]
        chunks_by_index.setdefault(idx, []).append(
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
        "tool_name": tool_name,
        "mode": params.mode,
        "candidates_evaluated": result.candidates_evaluated,
        "sources_with_hits": result.sources_with_hits,
        "sources_total": result.sources_total,
        "dropped_by_gate": result.dropped_by_gate,
        "reranked": result.reranked,
        "gate_margin": result.gate_margin,
        "neighbors_pulled": result.neighbors_pulled,
        "scoped_pool_size": pool_size,
        "facet_scope": {
            "sequence_number": sequence_number,
            "season_number": season_number,
            "matched_count": facet_matched_count,
            "no_match": facet_no_match,
        },
        "source_coverage": [
            {
                "source_id": source_map.get(s.video_id, ""),
                "video_id": s.video_id,
                "title": s.video_title,
            }
            for s in result.source_coverage
        ],
        "_chunks": (
            (f"{_NO_COVERAGE_NOTE}\n\n" if not result.chunks else "")
            + (f"{_NO_MATCH_NOTE}\n\n" if facet_no_match else "")
            + f"Sources retrieved this turn: {', '.join(f'[{i}]' for i in turn_indices)}. "
            "Cite only these indices.\n\n"
            f"{_build_source_headers(registry)}\n\n" + _build_fenced_chunks(chunks_by_index, registry)
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
    if name in RETRIEVE_TOOL_NAMES:
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
    if tool_name in RETRIEVE_TOOL_NAMES:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError(f"{tool_name} requires a non-empty 'query' string, got {query!r}")

        sequence_number = _facet_int(arguments.get("sequence_number"), "sequence_number")
        season_number = _facet_int(arguments.get("season_number"), "season_number")

        # Non-scoped tools ignore facet args even if present (defensive — LLM may misroute).
        if tool_name != RETRIEVE_SCOPED_TOOL.name and (sequence_number is not None or season_number is not None):
            logger.warning(
                "%s called with facet args (seq=%s season=%s); ignoring — only retrieve_scoped honors facets",
                tool_name,
                sequence_number,
                season_number,
            )
            sequence_number = None
            season_number = None

        logger.info(
            "retrieve tool call: tool=%s query=%r seq=%s season=%s",
            tool_name,
            query,
            sequence_number,
            season_number,
        )
        return await execute_retrieve(
            query=query,
            source_ids=source_ids,
            cfg=cfg,
            tool_name=tool_name,
            registry=registry,
            source_map=source_map,
            sequence_number=sequence_number,
            season_number=season_number,
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


def _summarize_stale_retrieve_block(block: dict) -> str:
    """Compact replacement for a prior-turn retrieve tool_result.

    Prior-turn raw excerpts pollute the current turn: the LLM treats stale
    chunks the same as fresh results and regenerates them, and survey query
    expansion borrows stale entities. Replay only a salient tag (query +
    which sources) so the LLM knows what was retrieved before without the
    raw text. To reuse the content, it must retrieve again this turn.
    """
    summary = block.get("result", {}).get("summary", {})
    query = summary.get("query") or block.get("arguments", {}).get("query", "")
    coverage = summary.get("source_coverage", []) or []
    chunks = block.get("result", {}).get("chunks", []) or []
    titles = ", ".join(f'"{c.get("title", "")}"' for c in coverage) or "(none)"
    return (
        f'[Prior-turn retrieval — query: "{query}"] '
        f"Retrieved {len(chunks)} excerpts from: {titles}. "
        "Excerpt text omitted to avoid stale-context contamination; "
        "call retrieve / survey / retrieve_scoped again if this turn needs the content."
    )


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
            result_payload = _summarize_stale_retrieve_block(b) if name in RETRIEVE_TOOL_NAMES else json.dumps(result)
            tool_result_content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": result_payload})
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
            result_payload = _summarize_stale_retrieve_block(b) if name in RETRIEVE_TOOL_NAMES else json.dumps(result)
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result_payload,
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
            if block.get("name") not in RETRIEVE_TOOL_NAMES:
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
