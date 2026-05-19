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


# Messages shorter than 3 non-whitespace chars, pure digits, or matching these
# stopwords get the trivial-acknowledgment path (#272 fix).
_TRIVIAL_STOPWORDS = frozenset(
    {
        "嗯",
        "ok",
        "好的",
        "好",
        "是",
        "对",
        "yes",
        "no",
        "thanks",
        "谢谢",
        "k",
        "kk",
    }
)

_TRIVIAL_PATH_NOTE = "The user sent a short acknowledgment. Respond naturally without calling retrieve."


def trivial_ack_note(message: str) -> str | None:
    """Return a system note when `message` is a trivial acknowledgment.

    Short (<3 non-whitespace chars), pure-digit, or stopword messages
    ("嗯"/"ok"/"thanks"/...) are conversation-control, not content queries —
    return _TRIVIAL_PATH_NOTE so the caller strips replayed retrieve blocks
    and instructs the LLM not to call retrieve. Returns None otherwise
    (pass-through: prior excerpts stay in context, LLM self-judges reuse
    from the prompt instruction in build_grounding_prompt).
    """
    stripped = message.strip()
    non_ws = "".join(stripped.split())
    if len(non_ws) < 3:
        logger.info("trivial_ack_note → trivial (short, len=%d)", len(non_ws))
        return _TRIVIAL_PATH_NOTE
    if non_ws.isdigit():
        logger.info("trivial_ack_note → trivial (digits)")
        return _TRIVIAL_PATH_NOTE
    if stripped.lower() in _TRIVIAL_STOPWORDS:
        logger.info("trivial_ack_note → trivial (stopword=%r)", stripped.lower())
        return _TRIVIAL_PATH_NOTE
    return None


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


DEFAULT_EXPECTED_HITS = "few"


_PARAMS_BY_HITS = {
    "one": RetrievalParams(depth_per_source=1, top_k=2, expected_hits="one"),
    "few": RetrievalParams(depth_per_source=2, top_k=8, expected_hits="few"),
    "many": RetrievalParams(depth_per_source=5, top_k=24, expected_hits="many"),
}


def params_for_expected_hits(expected_hits: str) -> RetrievalParams:
    """Map expected_hits to RetrievalParams. Defaults to 'few' for unknown values."""
    return _PARAMS_BY_HITS.get(expected_hits, _PARAMS_BY_HITS[DEFAULT_EXPECTED_HITS])


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
        "Retrieve information from video transcripts. Searches all sources "
        "by default.\n\n"
        "If the user's question explicitly references an episode / part or a "
        "season number (e.g. 第八集, 'part 3', 第二季), pass sequence_number / "
        "season_number — the backend scopes the search to matching sources. "
        "Omit them otherwise (omission searches all sources).\n\n"
        "Use expected_hits='one' for single-fact questions, 'few' (default) "
        "for narrow content questions, 'many' for survey/summary questions.\n\n"
        "Do NOT use for pure greetings or conversation-control messages.\n\n"
        "Excerpts you already retrieved remain in the conversation as tool "
        "results. If they already answer the question, cite them directly — "
        "do not call retrieve again. Call retrieve only when the conversation "
        "lacks the needed excerpts.\n\n"
        "Examples:\n"
        "  # General question — searches all sources\n"
        '  retrieve(query="长期情景记忆如何保存")\n\n'
        "  # User scoped to an episode\n"
        '  retrieve(query="第八集主要讲什么", sequence_number=8)\n\n'
        "  # User scoped to a season\n"
        '  retrieve(query="第二季讲了什么", season_number=2)'
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — key terms or question in the user's language",
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
            "sequence_number": {
                "type": "integer",
                "description": (
                    "Episode / part number, ONLY if the user's question explicitly "
                    "references it — e.g. 第八集 → sequence_number=8, 'part 3' → 3. "
                    "Omit if not mentioned (omission searches all sources)."
                ),
            },
            "season_number": {
                "type": "integer",
                "description": (
                    "Season number, ONLY if the user's question explicitly references "
                    "it — e.g. 第二季 → season_number=2. "
                    "Omit if not mentioned (omission searches all sources)."
                ),
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
    registry: dict[str, CitationRegistryEntry] | None = None,
    source_map: dict[str, str] | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
    expected_hits: str = DEFAULT_EXPECTED_HITS,
) -> dict:
    if registry is None:
        registry = {}
    if source_map is None:
        source_map = {}

    params = params_for_expected_hits(expected_hits)
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
    # Also populate CitationRegistryEntry fields from the first chunk per source.
    for c in result.chunks:
        sid = source_map.get(c.video_id)
        if sid and sid in registry:
            cid = f"{c.video_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}"
            registry[sid].chunk_ids.add(cid)
            # Populate entry fields from the first chunk for this source.
            # Content arrives already stripped (no [N] markers).
            entry = registry[sid]
            if entry.timestamp_start is None:
                entry.first_chunk_id = cid
                entry.timestamp_start = c.timestamp_start
                entry.timestamp_end = c.timestamp_end
                entry.rerank_score = c.score
                entry.preview = c.content

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
        "dropped_by_gate": result.dropped_by_gate,
        "reranked": result.reranked,
        "gate_margin": result.gate_margin,
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
            (f"{_NO_MATCH_NOTE}\n\n" if facet_no_match else "")
            + f"Sources retrieved this turn: {', '.join(f'[{i}]' for i in turn_indices)}. "
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
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError(f"retrieve requires a non-empty 'query' string, got {query!r}")

        sequence_number = _facet_int(arguments.get("sequence_number"), "sequence_number")
        season_number = _facet_int(arguments.get("season_number"), "season_number")

        logger.info(
            "retrieve tool call: query=%r seq=%s season=%s expected_hits=%r",
            query,
            sequence_number,
            season_number,
            arguments.get("expected_hits", DEFAULT_EXPECTED_HITS),
        )
        return await execute_retrieve(
            query=query,
            source_ids=source_ids,
            cfg=cfg,
            registry=registry,
            source_map=source_map,
            sequence_number=sequence_number,
            season_number=season_number,
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
