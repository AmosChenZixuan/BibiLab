"""RAG ledger reconstruction for a finished chat turn.

Extracted from routers/chat.py, which stays the HTTP + SSE surface. Pure
in-memory transformation over a turn's retrieval bookkeeping — no I/O, no
awaits, no dependence on the stream's lifetime.
"""

from bibilab.pipeline.chat_tools import CitationRegistryEntry


def build_rag_ledger(
    *,
    retrieve_calls: list[dict],
    read_section_calls: list[dict],
    content_blocks: list[dict],
    citation_registry: dict[str, CitationRegistryEntry],
) -> list[dict]:
    """Narrow retrieve coverage to cited sections and rebuild context[] from the registry.

    Call once per turn, after the last delta: `content_blocks` must already
    hold every citation block the turn emitted, since those are what decides
    which sections survive. Mutates the call dicts in place and returns the
    concatenated ledger (retrieve calls first) in the shape persisted on the
    message and emitted as the final `rag` SSE event.
    """
    if retrieve_calls:
        emitted_indices = {cb["index"] for cb in content_blocks if cb.get("type") == "citation"}
        if emitted_indices:
            emitted_section_ids = {sid for sid, entry in citation_registry.items() if entry.index in emitted_indices}
        else:
            emitted_section_ids = set()
        for call in retrieve_calls:
            # A turn that cited nothing keeps its coverage un-narrowed: filtering
            # by an empty set would erase a ledger the user can still inspect.
            if emitted_section_ids:
                call["section_coverage"] = [
                    s for s in call["section_coverage"] if s.get("section_id") in emitted_section_ids
                ]
            # One context entry per section left in section_coverage, narrowed or full.
            # Ordered, not a set: set iteration follows string hash order, which Python
            # randomizes per process, so context[] would reshuffle between restarts.
            section_ids_in_call = dict.fromkeys(s["section_id"] for s in call["section_coverage"])
            context_entries = []
            for sid in section_ids_in_call:
                entry = citation_registry.get(sid)
                if entry is not None:
                    context_entries.append(
                        {
                            "section_id": sid,
                            "section_seq": entry.seq,
                            "chunk_id": entry.first_chunk_id,
                            "citation_index": entry.index,
                            "source_id": entry.source_id,
                            "source_title": entry.title,
                            "timestamp_start": entry.timestamp_start,
                            "timestamp_end": entry.timestamp_end,
                            "rerank_score": entry.rerank_score,
                            "preview": entry.preview,
                        }
                    )
            call["context"] = context_entries

    for rs in read_section_calls:
        # The registry is keyed by section_id and nothing else — a row's
        # source_id names nothing in it.
        section_id = rs.get("section_id", "")
        entry = citation_registry.get(section_id) if section_id else None
        if entry is not None and not rs.get("source_title"):
            rs["source_title"] = entry.title or ""
        # read_section rows carry no chunk context — the read is bounded
        # verbatim transcript, not a fenced locator result. A synthetic entry
        # with zeroed fields would render as "0:00 / 0.00" in the frontend
        # ledger; an empty array lets the renderer branch on tool_name and
        # show a "read in full" affordance instead.
        rs["context"] = []

    return retrieve_calls + read_section_calls
