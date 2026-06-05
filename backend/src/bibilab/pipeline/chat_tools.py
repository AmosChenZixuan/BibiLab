"""Tool definitions and execution for chat."""

import json
import logging
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import (
    get_segments_for_ranges,
    get_source,
    get_source_facets,
    get_transcript_segments,
)
from bibilab.pipeline._shared import ToolDefinition
from bibilab.pipeline.digest import parse_facet_int
from bibilab.pipeline.embed import retrieve
from bibilab.pipeline.transcribe import WhisperSegment, build_speaker_namespace, format_turns

logger = logging.getLogger(__name__)

# Prepended to the LLM-visible find_passages result when facet scoping found
# no matching source and failed open to the full pool. Fact-only (#16.7 lock):
# the directive (say-so-before-answering) is owned by the prompt, not the
# tool. The fact is the LLM's sole degraded-scope signal; the directive moves
# to #372 grounding prompt. English by design: _chunks is LLM-facing.
_NO_MATCH_NOTE = "No source matched the requested episode/season; searched all sources instead."


@dataclass
class CitationRegistryEntry:
    index: int
    source_id: str
    title: str = ""
    chunk_ids: set[str] = field(default_factory=set)
    # Populated at SSE-build time from execute_find_passages chunk data.
    # Used to reconstruct context[] for persisted metadata.
    first_chunk_id: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    rerank_score: float | None = None
    preview: str | None = None


def strip_internal(result: dict) -> dict:
    """Drop `_`-prefixed internal keys (e.g. `_chunks`) from a tool result.

    Internal keys are LLM-facing formatting / private metadata; they must not
    reach the client SSE payload or the persisted tool_block. Public keys are
    those NOT prefixed with `_`.
    """
    return {k: v for k, v in result.items() if not k.startswith("_")}


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


def _partition_unseen_chunks(chunks: list, seen_chunk_ids: set[str]) -> list:
    """Return the chunks not already shown this turn, recording their ids.

    chunk_id = ``{source_id}_{int(timestamp_start)}_{int(timestamp_end)}`` —
    the same key used for citation tracking (chat_tools.py:456/478). Dedup is
    turn-scoped: parallel and multi-hop find_passages calls share one set so a
    given chunk is rendered to the LLM at most once per turn. Mutates
    seen_chunk_ids in place.
    """
    new = []
    for c in chunks:
        cid = f"{c.source_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}"
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        new.append(c)
    return new


# Tool defs + registry for the RAG v2 two-tool surface (#370/#371).
# RETRIEVE_TOOL_NAMES is the set whose tool results carry a replayable chunk
# array (only find_passages). read_source shares the SAME citation registry and
# allocates a new [N] when find_passages has not already registered the source
# this turn (dedup by source_id — spec §5.7).
FIND_PASSAGES_TOP_K = 8

FIND_PASSAGES_TOOL = ToolDefinition(
    name="find_passages",
    description=(
        "Search the video transcripts for excerpts relevant to a question. "
        "Returns the most relevant passages across all sources (or a single "
        "episode/season if you pass a facet), each fenced under its source with "
        "a [N] citation index.\n\n"
        "This is a LOCATOR: it surfaces *which sources* are relevant and gives "
        "you fragments to read. The fragments may not fully answer the question. "
        "If they answer it, synthesize. If a source is clearly on-topic but the "
        "fragments miss the specific thing asked, call read_source on that source "
        "to read it in full.\n\n"
        "Pass the user's question in natural form, copying proper nouns verbatim. "
        "Pass sequence_number / season_number ONLY when the current message "
        "explicitly names an episode (第八集) or season (第二季).\n\n"
        "Examples:\n"
        '  find_passages(query="演讲中提到的主要观点")\n'
        '  find_passages(query="第三个论据的具体内容", sequence_number=2)'
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user's question in natural form."},
            "sequence_number": {
                "type": "integer",
                "description": "Episode / part number — only if explicitly named in the CURRENT message",
            },
            "season_number": {
                "type": "integer",
                "description": "Season number — only if explicitly named in the CURRENT message",
            },
        },
        "required": ["query"],
    },
)

READ_SOURCE_TOOL = ToolDefinition(
    name="read_source",
    description=(
        "Read ONE source's full continuous transcript (summary + timestamped "
        "narrative). Expensive — use only when find_passages shows a source is "
        "on-topic but its fragments miss the specific thing asked, or for "
        "narrative-coverage questions about a named episode/season.\n\n"
        "Resolves to EXACTLY ONE source. Pass source_id (from a find_passages "
        "result) OR a facet (sequence_number / season_number). If a facet matches "
        "multiple sources it errors with the candidates — narrow it or pass "
        "source_id. For 'what does season N cover', issue parallel read_source "
        "calls (one per episode), not one fan-out call.\n\n"
        "Examples:\n"
        "  read_source(sequence_number=5)\n"
        '  read_source(source_id="…from a find_passages result…")'
    ),
    parameters={
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "description": "Exact source id, e.g. from a find_passages result."},
            "sequence_number": {"type": "integer", "description": "Episode / part number."},
            "season_number": {"type": "integer", "description": "Season number."},
        },
        "required": [],
    },
)

RETRIEVE_TOOL_NAMES: frozenset[str] = frozenset({FIND_PASSAGES_TOOL.name})


async def _resolve_single_source(
    source_ids: list[str],
    source_id: str | None,
    sequence_number: int | None,
    season_number: int | None,
) -> tuple[str | None, str | None]:
    """Resolve exactly one source from the pool. Returns (source_id, error_msg).

    Strict cardinality contract (spec §5.2): 0 matches → error, >1 → ambiguous
    error with candidates. Errors are LLM-facing strings (English, like the
    fence body), returned in the tool result — never raised (a raise aborts the
    SSE stream).
    """
    if source_id is not None:
        if source_id in source_ids:
            return source_id, None
        return None, f"source_id {source_id!r} is not in this list."

    predicates = {
        k: v for k, v in (("sequence_number", sequence_number), ("season_number", season_number)) if v is not None
    }
    if not predicates:
        return None, "read_source needs a source_id, sequence_number, or season_number."

    try:
        facets = await get_source_facets(source_ids)
    except Exception:  # noqa: BLE001 - fail closed: an unresolved read is an error, not a guess
        logger.warning("read_source: get_source_facets failed", exc_info=True)
        return None, "Could not resolve the source (facet lookup failed); try a source_id."

    matched = _filter_sources_by_facets(source_ids, facets, predicates)
    if not matched:
        return None, f"No source matches {predicates}."
    if len(matched) > 1:
        cands = ", ".join(
            f"seq={facets[m].get('sequence_number')} season={facets[m].get('season_number')}" for m in matched
        )
        return None, (
            f"Ambiguous — {len(matched)} sources match {predicates}; "
            f"specify season_number or source_id. Candidates: [{cands}]"
        )
    return matched[0], None


def _filter_sources_by_facets(
    source_ids: list[str],
    facets: dict[str, dict[str, int | None]],
    predicates: dict[str, int],
) -> list[str]:
    """Return source_ids whose stored facets match all (k, v) in predicates.

    Shared by _resolve_single_source (fail-closed) and execute_find_passages
    (fail-open); the cardinality policy lives at the call site.
    """
    return [sid for sid in source_ids if sid in facets and all(facets[sid].get(k) == v for k, v in predicates.items())]


def _build_source_narrative(source: dict, segments: list, idx: int) -> str:
    """Single source header + continuous timestamped transcript (spec §5.6).

    Reuses format_turns (include_time) for the speaker-turn body — NOT per-chunk
    fenced. Narrative continuity is the point.
    """
    # Empty-transcript edge (spec §5.6 / §16.3): SUPPRESS the header — it only
    # frames a body, and with no body it just hands the LLM fabrication fuel (a
    # title it already has from the fence). Return the explicit fact ONLY.
    if not segments:
        return (
            f"source [{idx}] has no transcript available "
            "(it may still be processing, or transcription may have failed)."
        )
    dur = source.get("duration_seconds")
    dur_line = f"Duration: {int(dur) // 60}:{int(dur) % 60:02d}, " if dur else ""
    header = (
        f'===== Source [{idx}]: "{source.get("title", "")}"\n'
        f"Summary: {source.get('summary') or '(none)'}\n"
        f"{dur_line}Language: {source.get('language') or 'unknown'}\n"
        f"====="
    )
    whisper_segs = [
        WhisperSegment(start=s["start_s"], end=s["end_s"], text=s["text"], speaker=s["speaker"]) for s in segments
    ]
    body = format_turns(whisper_segs, include_time=True)
    return header + "\n\n" + body


async def execute_read_source(
    source_ids: list[str],
    source_id: str | None,
    sequence_number: int | None,
    season_number: int | None,
    registry: dict[str, CitationRegistryEntry] | None = None,
) -> dict:
    if registry is None:
        registry = {}

    resolved, error = await _resolve_single_source(source_ids, source_id, sequence_number, season_number)
    if error is not None:
        logger.info(
            "read_source: unresolved (%s) source_id=%r seq=%s season=%s",
            error,
            source_id,
            sequence_number,
            season_number,
        )
        return {"_chunks": error, "source_id": None, "source_title": "", "tool_name": "read_source"}

    source = await get_source(resolved)
    if source is None:
        logger.info("read_source: resolved=%s but source row missing", resolved)
        return {
            "_chunks": f"source {resolved!r} not found.",
            "source_id": None,
            "source_title": "",
            "tool_name": "read_source",
        }
    # aiosqlite.Row supports [] but not .get; the narrative builder wants the
    # latter. Convert once at the boundary rather than wrapping every lookup.
    source = {k: source[k] for k in source.keys()}

    segments = await get_transcript_segments(resolved)

    # Shared registry, dedup by source_id (spec §5.7): reuse the existing [N] if
    # find_passages already registered this source this turn; else allocate next.
    entry = registry.get(resolved)
    if entry is None:
        next_index = max((e.index for e in registry.values()), default=0) + 1
        entry = CitationRegistryEntry(index=next_index, source_id=resolved, title=source.get("title", ""))
        registry[resolved] = entry

    logger.info(
        "read_source: resolved=%s seq=%s season=%s segments=%d idx=%d",
        resolved,
        sequence_number,
        season_number,
        len(segments),
        entry.index,
    )
    narrative = _build_source_narrative(source, segments, idx=entry.index)
    return {
        "_chunks": narrative,
        "source_id": resolved,
        "source_title": source["title"],
        "tool_name": "read_source",
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


async def execute_find_passages(
    query: str,
    source_ids: list[str],
    cfg: BibilabConfig,
    registry: dict[str, CitationRegistryEntry] | None = None,
    sequence_number: int | None = None,
    season_number: int | None = None,
    seen_chunk_ids: set[str] | None = None,
) -> dict:
    if seen_chunk_ids is not None and not isinstance(seen_chunk_ids, set):
        raise TypeError(f"seen_chunk_ids must be a set or None, got {type(seen_chunk_ids).__name__}")
    if registry is None:
        registry = {}

    pool_size = len(source_ids)

    # Deterministic facet scoping (#309). Facet matching is the sole
    # pre-retrieval narrowing. Fail-open: zero match (or facet-subquery DB
    # error) → full pool, never empty.
    facet_predicates = {
        k: v for k, v in (("sequence_number", sequence_number), ("season_number", season_number)) if v is not None
    }
    facet_matched_count: int | None = None
    facet_no_match = False
    scoped_source_ids: list[str] | None = None
    if facet_predicates:
        try:
            facets = await get_source_facets(source_ids)
        except Exception:  # noqa: BLE001
            logger.warning("find_passages: get_source_facets failed, fail-open to full pool", exc_info=True)
            facets = {}
        matched = _filter_sources_by_facets(source_ids, facets, facet_predicates)
        facet_matched_count = len(matched)
        if matched:
            scoped_source_ids = matched
        else:
            facet_no_match = True
            logger.warning(
                "find_passages: facet %s matched 0 sources, fail-open to full pool",
                facet_predicates,
            )

    logger.info(
        "find_passages dispatch: query=%r pool_size=%d",
        query,
        pool_size,
    )
    result = await retrieve(
        query_text=query,
        source_ids=source_ids,
        cfg=cfg,
        top_k=FIND_PASSAGES_TOP_K,
        scoped_source_ids=scoped_source_ids,
    )

    # Intra-turn dedup: drop chunks already rendered to the LLM this turn
    # (parallel / multi-hop find_passages can overlap). source_coverage and the
    # citation registry stay on the full result — a re-hit source keeps its [N].
    if seen_chunk_ids is None:
        new_chunks = result.chunks
    else:
        new_chunks = _partition_unseen_chunks(result.chunks, seen_chunk_ids)
        deduped = len(result.chunks) - len(new_chunks)
        if deduped:
            logger.info(
                "find_passages_intraturn_dedup query=%r deduped=%d kept=%d",
                query,
                deduped,
                len(new_chunks),
            )
    dedup_all_shown = bool(result.chunks) and not new_chunks

    # Assign indices: new sources get next available index
    next_index = max((e.index for e in registry.values()), default=0) + 1
    for s in result.source_coverage:
        sid = s.source_id
        if sid not in registry:
            registry[sid] = CitationRegistryEntry(
                index=next_index,
                source_id=sid,
                title=s.video_title,
            )
            next_index += 1

    # Build source_id → registry index lookup for chunk formatting. Every sid
    # in source_coverage was just added to the registry above, so the membership
    # check is a no-op — read directly.
    source_id_to_index = {s.source_id: registry[s.source_id].index for s in result.source_coverage}

    # Reconstruct the speaker-turn body for each displayed chunk (kept from v1).
    displayed = [c for c in new_chunks if c.source_id in source_id_to_index]
    ranges = [
        (c.source_id, c.seg_start, c.seg_end) for c in displayed if c.seg_start is not None and c.seg_end is not None
    ]
    seg_rows = await get_segments_for_ranges(ranges) if ranges else []
    segs_by_source: dict[str, list[WhisperSegment]] = {}
    for r in seg_rows:
        segs_by_source.setdefault(r["source_id"], []).append(
            WhisperSegment(start=r["start_s"], end=r["end_s"], text=r["text"], speaker=r["speaker"])
        )
    ns_by_source = {sid: build_speaker_namespace(segs) for sid, segs in segs_by_source.items()}

    def _turn_body(c):
        if c.seg_start is None or c.seg_end is None or c.source_id not in segs_by_source:
            return c.content
        idx = source_id_to_index[c.source_id]
        rows = [r for r in seg_rows if r["source_id"] == c.source_id and c.seg_start <= r["seq"] <= c.seg_end]
        chunk_segs = [
            WhisperSegment(start=r["start_s"], end=r["end_s"], text=r["text"], speaker=r["speaker"]) for r in rows
        ]
        if not chunk_segs:
            return c.content
        return format_turns(
            chunk_segs, include_time=True, citation_index=idx, speaker_namespace=ns_by_source[c.source_id]
        )

    body_by_chunk = {id(c): _turn_body(c) for c in displayed}

    # Entry-level fields seeded from the first chunk per source.
    for c in new_chunks:
        sid = c.source_id
        if sid in registry:
            cid = f"{sid}_{int(c.timestamp_start)}_{int(c.timestamp_end)}"
            registry[sid].chunk_ids.add(cid)
            entry = registry[sid]
            if entry.timestamp_start is None:
                entry.first_chunk_id = cid
                entry.timestamp_start = c.timestamp_start
                entry.timestamp_end = c.timestamp_end
                entry.rerank_score = c.score
                entry.preview = body_by_chunk.get(id(c), c.content)

    turn_indices = sorted(set(source_id_to_index.values()))

    chunks_by_index: dict[int, list[str]] = {}
    raw_chunks = []
    for c in new_chunks:
        if c.source_id not in source_id_to_index:
            continue
        idx = source_id_to_index[c.source_id]
        chunks_by_index.setdefault(idx, []).append(body_by_chunk[id(c)])
        raw_chunks.append(
            {
                "source_id": c.source_id,
                "chunk_id": f"{c.source_id}_{int(c.timestamp_start)}_{int(c.timestamp_end)}",
                "content": c.content,
                "video_title": c.video_title,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
                "citation_index": idx,
            }
        )

    # The all-directive "no-coverage" note is DELETED — that behavior lives in
    # the grounding prompt (#372). The tool emits only the FACT: empty pool.
    no_coverage_fact = "" if result.chunks else "find_passages found no relevant excerpts for this query.\n\n"
    no_match_fact = f"{_NO_MATCH_NOTE}\n\n" if facet_no_match else ""

    return {
        "query": query,
        "tool_name": FIND_PASSAGES_TOOL.name,
        "candidates_evaluated": result.candidates_evaluated,
        "sources_with_hits": result.sources_with_hits,
        "sources_total": result.sources_total,
        "reranked": result.reranked,
        "scoped_pool_size": pool_size,
        "facet_scope": {
            "sequence_number": sequence_number,
            "season_number": season_number,
            "matched_count": facet_matched_count,
            "no_match": facet_no_match,
        },
        "source_coverage": [
            {
                "source_id": s.source_id,
                "title": s.video_title,
            }
            for s in result.source_coverage
        ],
        "_chunks": (
            no_coverage_fact
            + no_match_fact
            + (
                "All matching passages for this query were already retrieved earlier this turn.\n\n"
                if dedup_all_shown
                else (
                    f"Sources retrieved this turn: {', '.join(f'[{i}]' for i in turn_indices)}. "
                    "Cite only these indices.\n\n"
                    f"{_build_source_headers(registry)}\n\n" + _build_fenced_chunks(chunks_by_index, registry)
                )
            )
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

    For retrieve-family, internal underscore-prefixed fields are stripped and
    raw chunk snapshots are attached so replay survives re-embedding. For
    other tools (read_source), underscore-prefixed fields are also stripped —
    read_source's narrative is 30-150K tokens and is never replayed or reseeded,
    so persisting it is pure DB bloat (spec §16.7).
    """
    summary = strip_internal(result)
    if name in RETRIEVE_TOOL_NAMES:
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
        "result": summary,
    }


async def execute_tool(
    tool_name: str,
    arguments: dict,
    source_ids: list[str],
    cfg: BibilabConfig,
    registry: dict[str, CitationRegistryEntry] | None = None,
    seen_chunk_ids: set[str] | None = None,
) -> dict:
    if tool_name == FIND_PASSAGES_TOOL.name:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError(f"find_passages requires a non-empty 'query' string, got {query!r}")
        sequence_number = _facet_int(arguments.get("sequence_number"), "sequence_number")
        season_number = _facet_int(arguments.get("season_number"), "season_number")
        return await execute_find_passages(
            query=query,
            source_ids=source_ids,
            cfg=cfg,
            registry=registry,
            sequence_number=sequence_number,
            season_number=season_number,
            seen_chunk_ids=seen_chunk_ids,
        )
    if tool_name == READ_SOURCE_TOOL.name:
        return await execute_read_source(
            source_ids=source_ids,
            source_id=arguments.get("source_id"),
            sequence_number=_facet_int(arguments.get("sequence_number"), "sequence_number"),
            season_number=_facet_int(arguments.get("season_number"), "season_number"),
            registry=registry,
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

    v2 rule (spec §5.5): drop retrieve-family and read_source tool exchanges
    entirely from cross-turn replay — the LLM may rely on the assistant prose
    but not the prior tool result (stale-context contamination).
    """
    blocks = msg.get("tool_blocks")
    if not blocks:
        # Strip tool_blocks if present-but-empty; producer expects clean shape.
        clean = {k: v for k, v in msg.items() if k != "tool_blocks"}
        return [clean]

    # v2: drop retrieve + read_source tool exchanges from cross-turn replay.
    blocks = [b for b in blocks if b.get("name") not in (RETRIEVE_TOOL_NAMES | {READ_SOURCE_TOOL.name})]
    if not blocks:
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
            result_payload = json.dumps(b.get("result"))
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
            result_payload = json.dumps(b.get("result"))
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
