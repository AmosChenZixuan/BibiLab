"""Tool definitions and execution for chat."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from bibilab.config import BibilabConfig
from bibilab.db import (
    get_sections,
    get_segments_for_ranges,
    get_source,
    get_source_facets,
    rows_to_segments,
)
from bibilab.pipeline._shared import ToolDefinition, format_hms
from bibilab.pipeline.digest import parse_facet_int
from bibilab.pipeline.embed import retrieve
from bibilab.pipeline.transcribe import WhisperSegment, build_speaker_namespace, format_turns

logger = logging.getLogger(__name__)

# Prepended to the LLM-visible find_passages result when facet scoping found
# no matching source and failed open to the full pool. Fact-only:
# the directive (say-so-before-answering) is owned by the prompt, not the
# tool. The fact is the LLM's sole degraded-scope signal; the directive moves
# to the grounding prompt. English by design: _chunks is LLM-facing.
_NO_MATCH_NOTE = "No source matched the requested episode/season; searched all sources instead."


@dataclass
class CitationRegistryEntry:
    index: int
    # section_id is typed str but is in fact the INTEGER sections.id row PK
    # (sourced from `get_sections`'s r["id"], then threaded through
    # _alloc_section's first arg). The SSE event serializes it as a JSON
    # number; the FE's ContentBlock.citation.section_id declares str, and
    # chat-utils.coerceCitationEvent normalizes number→string at the SSE
    # boundary so strict-equality jumps (SourcesViewerMode.resolveTargetIdx)
    # don't fall through to the timestampStart branch. Keep this as str to
    # match the dict-key type in `registry`; the FE coercion is the fix.
    section_id: str
    source_id: str
    title: str = ""
    seq: int | None = None  # 1-based section seq within the source
    citable: bool = False  # True if the section's [N] is a clickable citation in the chat UI
    chunk_ids: set[str] = field(default_factory=set)
    # Populated at SSE-build time from execute_find_passages chunk data.
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


def _build_section_fence_header(entry: CitationRegistryEntry) -> str:
    return (
        f'===== [{entry.index}] "{entry.title}" · Section {entry.seq} '
        f"({format_hms(entry.timestamp_start)}–{format_hms(entry.timestamp_end)}) ====="
    )


# Rendered between two non-adjacent fragments of one section to signal that
# transcript was elided between them. Adjacent fragments join on a plain
# newline (continuous speech); a gap marker means the model must not read
# across the boundary as if it were continuous.
_FRAGMENT_GAP = "[…]"


def _join_section_fragments(fragments: list[tuple[int, int, str]]) -> str:
    """Join a section's chunk fragments in chronological (segment) order.

    Each fragment is (seg_start, seg_end, body). Retrieval returns chunks in
    rerank order, so a section's fragments can arrive out of spoken order;
    sorting by seg_start restores it. Chunk ranges are disjoint and nest within
    a section, so the sorted seg_end is monotonic. Between two fragments that are
    not seg-adjacent (a gap in segment indices ⇒ skipped transcript) a
    `_FRAGMENT_GAP` marker is inserted so the elision reads as a gap, not as
    continuous speech.
    """
    if not fragments:
        return ""
    ordered = sorted(fragments, key=lambda f: f[0])
    out = ordered[0][2]
    prev_end = ordered[0][1]
    for seg_start, seg_end, body in ordered[1:]:
        out += (f"\n\n{_FRAGMENT_GAP}\n\n" if seg_start > prev_end + 1 else "\n") + body
        prev_end = seg_end
    return out


def _build_fenced_sections(
    chunks_by_index: dict[int, list[tuple[int, int, str]]],
    summaries_by_index: dict[int, str],
    registry: dict[str, CitationRegistryEntry],
) -> str:
    """Render each surfaced section: fence header, then its summary, then its
    chunk fragments (if any). Emitted in ascending [N] order. A section with a
    summary but no chunks (outline-only) renders header + summary only.
    """
    by_index = {e.index: e for e in registry.values()}
    blocks: list[str] = []
    for idx in sorted(set(chunks_by_index) | set(summaries_by_index)):
        entry = by_index.get(idx)
        if entry is None:
            continue
        parts = [_build_section_fence_header(entry)]
        summary = summaries_by_index.get(idx)
        if summary:
            parts.append(summary)
        body = _join_section_fragments(chunks_by_index.get(idx, []))
        if body:
            parts.append(body)
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _chunk_id(chunk) -> str:
    """Canonical chunk_id used by citation tracking, raw_chunks, and intra-turn
    dedup. Centralizing the formula here guarantees the dedup key can never
    drift from the citation key."""
    return f"{chunk.source_id}_{int(chunk.timestamp_start)}_{int(chunk.timestamp_end)}"


def _section_for_seg(sections: list[tuple[str, int, int, int]], seg_start: int) -> tuple[str, int, int, int] | None:
    """Return the (section_id, seq, seg_start, seg_end) tuple whose range
    contains `seg_start`, or None if no section contains it.

    Chunks nest in exactly one section (a chunk's [seg_start, seg_end] is fully
    contained in one section's range — CLAUDE.md sections invariant), so mapping
    by the chunk's seg_start is sufficient and unambiguous.
    """
    for sid, seq, s, e in sections:
        if s <= seg_start <= e:
            return (sid, seq, s, e)
    return None


def _partition_unseen_chunks(chunks: list, seen_chunk_ids: set[str]) -> list:
    """Return the chunks not already shown this turn, recording their ids.

    Dedup key is the canonical chunk_id (see ``_chunk_id``) — the same key
    used by citation tracking. Dedup is turn-scoped: parallel and multi-hop
    find_passages calls share one set so a given chunk is rendered to the LLM
    at most once per turn. Mutates seen_chunk_ids in place.
    """
    new = []
    for c in chunks:
        cid = _chunk_id(c)
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        new.append(c)
    return new


# Tool-name constants. These are the single source of truth for the wire
# string the LLM sees, what we record in tool_result envelopes, and what we
# filter in RETRIEVE_TOOL_NAMES. Renaming a tool means updating the
# constants — every other site already routes through them.
TOOL_NAME_FIND_PASSAGES = "find_passages"
TOOL_NAME_READ_SECTION = "read_section"

# Tool defs + registry for the RAG v2 two-tool surface.
# RETRIEVE_TOOL_NAMES is the set whose tool results carry a replayable chunk
# array (only find_passages). read_section resolves an already-registered [N]
# from the citation registry to one section's bounded verbatim transcript
# (no new index allocation).
FIND_PASSAGES_TOP_K = 8

FIND_PASSAGES_TOOL = ToolDefinition(
    name=TOOL_NAME_FIND_PASSAGES,
    description=(
        "Search the video transcripts for excerpts relevant to a question. "
        "Returns the most relevant passages across all sources (or a single "
        "episode/season if you pass a facet), each fenced under its source with "
        "a [N] citation index.\n\n"
        "This is a LOCATOR: it surfaces *which sections* are relevant and gives "
        "you fragments to read. The fragments may not fully answer the question. "
        "If they answer it, synthesize. If a section is clearly on-topic but the "
        "fragments miss the specific thing asked, call read_section on that [N] "
        "to read the section in full.\n\n"
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

READ_SECTION_TOOL = ToolDefinition(
    name=TOOL_NAME_READ_SECTION,
    description=(
        "Read ONE section's full verbatim transcript. A find_passages result "
        "fences each section under a [N] index; pass that index to read that "
        "section in full when its summary/fragments miss the specific thing "
        "asked. For 'what does episode N cover', read the outline summaries "
        "first; to quote a specific section, read_section that [N]. To read a "
        "whole episode verbatim, issue parallel read_section calls, one per "
        "section index.\n\n"
        'Example:\n  read_section(section_id="[5]")'
    ),
    parameters={
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": "Section citation index from a find_passages result, e.g. '[5]'.",
            },
        },
        "required": ["section_id"],
    },
)

RETRIEVE_TOOL_NAMES: frozenset[str] = frozenset({TOOL_NAME_FIND_PASSAGES})

# A bracketed [N] parses even with trailing text — the LLM copies the whole
# fence label ('[4] "title…"') it sees in find_passages results. A BARE index
# stays whole-string-anchored: a stray title like "Episode 5 discussion" must
# NOT silently parse to 5 (it would read the wrong section).
_CITATION_INDEX_RE = re.compile(r"^\s*(?:source\s*)?\[?\s*(\d+)\s*(?:\].*|)$", re.IGNORECASE)


def _filter_sources_by_facets(
    source_ids: list[str],
    facets: dict[str, dict[str, int | None]],
    predicates: dict[str, int],
) -> list[str]:
    """Return source_ids whose stored facets match all (k, v) in predicates.

    Used by execute_find_passages facet scoping (fail-open). Card policy lives
    at the call site.
    """
    return [sid for sid in source_ids if sid in facets and all(facets[sid].get(k) == v for k, v in predicates.items())]


async def _build_section_narrative(entry: CitationRegistryEntry) -> str:
    """One section's verbatim speaker-turn body, bounded by its seg range,
    rendered with the section's own citation index."""
    rows = await get_sections(entry.source_id)
    sec = next((r for r in rows if r["id"] == entry.section_id), None)
    if sec is None:
        return f"section [{entry.index}] not found."
    seg_rows = await get_segments_for_ranges([(entry.source_id, sec["seg_start"], sec["seg_end"])])
    if not seg_rows:
        # Empty-transcript edge: SUPPRESS the header — it only frames a body,
        # and with no body it just hands the LLM fabrication fuel (a title it
        # already has from the fence). Return the explicit fact ONLY.
        return (
            f"section [{entry.index}] has no transcript available "
            "(it may still be processing, or transcription may have failed)."
        )
    segs = rows_to_segments(seg_rows)
    ns = build_speaker_namespace(segs)
    header = f'===== [{entry.index}] "{entry.title}" · Section {entry.seq} ====='
    body = format_turns(segs, include_time=True, citation_index=entry.index, speaker_namespace=ns)
    return header + "\n\n" + body


def _read_section_error(msg: str) -> dict:
    """LLM-facing error envelope for execute_read_section. The shape matches
    the success return except the identity fields are None/empty — the parser
    + frontend expect a consistent return shape regardless of why the read
    failed, so the failure mode renders the same as a successful read with no
    section matched. Centralized so the parse-error and unknown-index branches
    stay in lockstep if a new field is added to the success return."""
    return {
        "_chunks": msg,
        "section_id": None,
        "source_id": None,
        "source_title": "",
        "tool_name": TOOL_NAME_READ_SECTION,
    }


async def execute_read_section(
    source_ids: list[str],
    section_id: str | int | None,
    registry: dict[str, CitationRegistryEntry] | None = None,
) -> dict:
    """Read ONE section's bounded verbatim transcript by [N] citation index.

    The index must already be registered this turn (via find_passages). The
    LLM-facing body uses format_turns with the section's own citation_index,
    so the speaker labels match the find_passages fence — citations stay
    bindable. The section stays citable (find_passages already registered it).
    Errors are LLM-facing strings, never raised.
    """
    registry = registry or {}
    # Resolve [N] → the section entry registered this turn.
    if isinstance(section_id, int):
        idx = section_id
    elif isinstance(section_id, str) and (m := _CITATION_INDEX_RE.match(section_id)):
        idx = int(m.group(1))
    else:
        return _read_section_error('section_id must be a citation index like "[5]".')
    entry = next((e for e in registry.values() if e.index == idx and e.source_id in source_ids), None)
    if entry is None:
        return _read_section_error(f"No section [{idx}] in this conversation. Call find_passages first.")
    narrative = await _build_section_narrative(entry)
    entry.citable = True
    logger.info("read_section: idx=%d section=%s source=%s", idx, entry.section_id, entry.source_id)
    return {
        "_chunks": narrative,
        "section_id": entry.section_id,
        "source_id": entry.source_id,
        "source_title": entry.title,
        "tool_name": TOOL_NAME_READ_SECTION,
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
    if registry is None:
        registry = {}

    pool_size = len(source_ids)

    # Deterministic facet scoping. Facet matching is the sole
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
    # (parallel / multi-hop find_passages can overlap). section_coverage and the
    # citation registry stay on the full result — a re-hit section keeps its [N].
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

    # Load sections for every source this turn that needs them — chunk-hit
    # sources (for the per-chunk section lookup) plus facet-scoped sources
    # (for the full outline expansion). Batched via asyncio.gather so a
    # turn touching many sources issues one DB roundtrip total, not N serial
    # roundtrips — and so the outline loop never re-queries sections the
    # chunk-hit path already loaded.
    hit_source_ids = {c.source_id for c in new_chunks}
    load_source_ids = set(hit_source_ids) | set(scoped_source_ids or [])
    sections_by_source: dict[str, list[tuple[str, int, int, int]]] = {}
    summary_by_section_id: dict[str, str] = {}
    ts_by_section_id: dict[str, tuple[float, float]] = {}
    if load_source_ids:
        rows_list = await asyncio.gather(*(get_sections(sid) for sid in load_source_ids))
        for sid, rows in zip(load_source_ids, rows_list):
            # seq is 0-based in the DB; the registry/fence display 1-based
            # (matches the "Section N" human label, see _build_section_fence_header).
            sections_by_source[sid] = [(r["id"], r["seq"] + 1, r["seg_start"], r["seg_end"]) for r in rows]
            for r in rows:
                summary_by_section_id[r["id"]] = r["summary"]
                ts_by_section_id[r["id"]] = (r["timestamp_start"], r["timestamp_end"])

    # Batched title fallback for sources the retrieval-result doesn't carry a
    # title for (typically scoped-but-not-hit sources — the outline loop needs
    # their title for the registry entry).
    title_by_source = {s.source_id: s.video_title for s in result.source_coverage}
    missing_titles = [s for s in (scoped_source_ids or []) if s not in title_by_source]
    if missing_titles:
        src_rows = await asyncio.gather(*(get_source(s) for s in missing_titles))
        for sid, row in zip(missing_titles, src_rows):
            title_by_source[sid] = row["title"] if row else ""

    # Reconstruct the speaker-turn body for each displayed chunk.
    # Section allocation happens below; the per-chunk turn body is built first
    # because the section's verbatim is the body, not just a header.
    ranges = [
        (c.source_id, c.seg_start, c.seg_end) for c in new_chunks if c.seg_start is not None and c.seg_end is not None
    ]
    seg_rows = await get_segments_for_ranges(ranges) if ranges else []
    all_segs = rows_to_segments(seg_rows)  # convert once; (row, seg) pairs stay aligned below
    segs_by_source: dict[str, list[WhisperSegment]] = {}
    for s, r in zip(all_segs, seg_rows):
        segs_by_source.setdefault(r["source_id"], []).append(s)
    ns_by_source = {sid: build_speaker_namespace(segs) for sid, segs in segs_by_source.items()}

    def _turn_body_for_index(c, idx: int) -> str:
        if c.seg_start is None or c.seg_end is None or c.source_id not in segs_by_source:
            return c.content
        chunk_segs = [
            s
            for s, r in zip(all_segs, seg_rows)
            if r["source_id"] == c.source_id and c.seg_start <= r["seq"] <= c.seg_end
        ]
        if not chunk_segs:
            return c.content
        return format_turns(
            chunk_segs, include_time=True, citation_index=idx, speaker_namespace=ns_by_source[c.source_id]
        )

    # next_index is closure-captured by _alloc_section; we update it once per
    # new section (avoids the O(n) max() walk that the old code did on every
    # _alloc_section call → O(n²) over the outline pass). Initialized to max+1
    # so the first new section in an empty registry gets index=1 (matching the
    # pre-closure behavior — old code did `max(...) + 1` inline per call).
    next_index = max((e.index for e in registry.values()), default=0) + 1

    def _alloc_section(section_id: str, source_id: str, title: str, seq: int) -> CitationRegistryEntry:
        nonlocal next_index
        entry = registry.get(section_id)
        if entry is None:
            entry = CitationRegistryEntry(
                index=next_index,
                section_id=section_id,
                source_id=source_id,
                title=title,
                seq=seq,
            )
            registry[section_id] = entry
            next_index += 1
        return entry

    # Map each displayed chunk → its section; allocate [N] per section.
    chunks_by_index: dict[int, list[tuple[int, int, str]]] = {}
    summaries_by_index: dict[int, str] = {}
    raw_chunks = []
    for c in new_chunks:
        sec = (
            _section_for_seg(sections_by_source.get(c.source_id, []), c.seg_start) if c.seg_start is not None else None
        )
        if sec is None:
            continue
        section_id, seq, _, _ = sec
        entry = _alloc_section(section_id, c.source_id, title_by_source.get(c.source_id, c.video_title), seq)
        idx = entry.index
        entry.citable = True
        cid = _chunk_id(c)
        entry.chunk_ids.add(cid)
        body = _turn_body_for_index(c, idx)
        # First-chunk seed mirrors the pre-section flow; later chunks expand
        # the section's [start, end] window.
        if entry.first_chunk_id is None:
            entry.first_chunk_id = cid
            entry.rerank_score = c.score
            entry.preview = body
        if entry.timestamp_start is None or c.timestamp_start < entry.timestamp_start:
            entry.timestamp_start = c.timestamp_start
        if entry.timestamp_end is None or c.timestamp_end > entry.timestamp_end:
            entry.timestamp_end = c.timestamp_end
        seg_end = c.seg_end if c.seg_end is not None else c.seg_start
        chunks_by_index.setdefault(idx, []).append((c.seg_start, seg_end, body))
        summaries_by_index[idx] = summary_by_section_id.get(section_id, "")
        raw_chunks.append(
            {
                "source_id": c.source_id,
                "section_id": section_id,
                "section_seq": seq,
                "chunk_id": cid,
                "content": c.content,
                "video_title": c.video_title,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
                "citation_index": idx,
            }
        )

    # Facet matched → emit the FULL section outline for each matched source:
    # register every section (summary, its own [N]). Outline-only sections
    # (no chunk hit) are first-class citations: citable from the start, with
    # the section summary attached as `preview` so the ledger row has a body.
    if scoped_source_ids:
        for sid in scoped_source_ids:
            title = title_by_source.get(sid, "")
            for section_id, seq, _, _ in sections_by_source.get(sid, []):
                entry = _alloc_section(section_id, sid, title, seq)
                # Outline-only sections have no chunk to derive timestamps from;
                # seed them from the section row so the fence header (and
                # persisted ledger) show the real span instead of 0:00–0:00.
                if entry.timestamp_start is None:
                    entry.timestamp_start, entry.timestamp_end = ts_by_section_id[section_id]
                if not entry.chunk_ids:
                    entry.citable = True
                    entry.preview = summary_by_section_id[section_id]
                summaries_by_index[entry.index] = summary_by_section_id[section_id]

    # Recompute turn_indices AFTER outline expansion so outline indices flow into section_coverage.
    turn_indices = sorted(set(chunks_by_index) | set(summaries_by_index))

    # The all-directive "no-coverage" note is DELETED — that behavior lives in
    # the grounding prompt. The tool emits only the FACT: empty pool.
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
        "section_coverage": [
            {
                "section_id": e.section_id,
                "source_id": e.source_id,
                "source_title": e.title,
                "seq": e.seq,
                "timestamp_start": ts[0],
                "timestamp_end": ts[1],
            }
            for e in sorted(registry.values(), key=lambda e: e.index)
            if e.index in turn_indices
            for ts in [ts_by_section_id.get(e.section_id, (None, None))]
        ],
        "_chunks": (
            no_coverage_fact
            + no_match_fact
            + (
                "All matching passages for this query were already retrieved earlier this turn.\n\n"
                if dedup_all_shown
                else _build_fenced_sections(chunks_by_index, summaries_by_index, registry)
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
    other tools (read_section), underscore-prefixed fields are also stripped —
    read_section's narrative is bounded to one section and is never replayed
    or reseeded, so persisting it is pure DB bloat.
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
    if tool_name == READ_SECTION_TOOL.name:
        return await execute_read_section(
            source_ids=source_ids,
            section_id=arguments.get("section_id"),
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

    v2 rule: drop retrieve-family and read_section tool exchanges
    entirely from cross-turn replay — the LLM may rely on the assistant prose
    but not the prior tool result (stale-context contamination).
    """
    blocks = msg.get("tool_blocks")
    if not blocks:
        # Strip tool_blocks if present-but-empty; producer expects clean shape.
        clean = {k: v for k, v in msg.items() if k != "tool_blocks"}
        return [clean]

    # v2: drop retrieve + read_section tool exchanges from cross-turn replay.
    blocks = [b for b in blocks if b.get("name") not in (RETRIEVE_TOOL_NAMES | {READ_SECTION_TOOL.name})]
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
            result_payload = json.dumps(result)
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
            result_payload = json.dumps(result)
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
    re-creates CitationRegistryEntry instances keyed by section_id so prior
    [N] markers in old assistant text continue to resolve to the same section.
    """
    for msg in history:
        for block in msg.get("tool_blocks") or []:
            if block.get("name") not in RETRIEVE_TOOL_NAMES:
                continue
            chunks = block.get("result", {}).get("chunks", [])
            for ch in chunks:
                section_id = ch.get("section_id")
                if not section_id:
                    continue
                ci = ch.get("citation_index")
                if ci is None:
                    continue
                entry = registry.get(section_id)
                if entry is None:
                    entry = CitationRegistryEntry(
                        index=ci,
                        section_id=section_id,
                        source_id=ch.get("source_id", ""),
                        title=ch.get("video_title", ""),
                        seq=ch.get("section_seq"),
                        citable=True,  # a persisted chunk means verbatim was shown
                    )
                    registry[section_id] = entry
                cid = ch.get("chunk_id")
                if cid:
                    entry.chunk_ids.add(cid)
