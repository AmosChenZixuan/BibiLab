"""Incremental citation parser — strips [N] tokens from LLM deltas, emits citation events."""

import json
import logging
import re

from bibilab.pipeline._shared import StreamEvent
from bibilab.pipeline.chat_tools import CitationRegistryEntry

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_delta(
    delta: str,
    buffer: str,
    index_to_entry: dict[int, CitationRegistryEntry],
) -> tuple[list[StreamEvent], str]:
    """Process a text delta through the citation parser.

    Returns (events, new_buffer). Events are stripped-text deltas and/or
    citation events with {index, source_id, chunk_ids} payloads.
    """
    buffer += delta
    events: list[StreamEvent] = []
    pos = 0

    for m in _CITATION_RE.finditer(buffer):
        start, end = m.span()
        n = int(m.group(1))

        if start > pos:
            events.append(StreamEvent(type="delta", content=buffer[pos:start]))

        entry = index_to_entry.get(n)
        if entry is not None:
            events.append(
                StreamEvent(
                    type="citation",
                    content=json.dumps(
                        {
                            "index": n,
                            "source_id": entry.source_id,
                            "chunk_ids": sorted(entry.chunk_ids),
                        }
                    ),
                )
            )
        else:
            logger.warning(
                "citation_hallucinated_index index=%d registry_size=%d",
                n,
                len(index_to_entry),
            )
            events.append(StreamEvent(type="delta", content=buffer[start:end]))

        pos = end

    # Keep trailing partial [N] in buffer for next delta
    after = buffer[pos:]
    last_bracket = after.rfind("[")
    if last_bracket != -1:
        after_bracket = after[last_bracket + 1 :]
        if after_bracket == "" or after_bracket.isdigit():
            if last_bracket > 0:
                events.append(StreamEvent(type="delta", content=after[:last_bracket]))
            new_buffer = after[last_bracket:]
        else:
            if after:
                events.append(StreamEvent(type="delta", content=after))
            new_buffer = ""
    else:
        if after:
            events.append(StreamEvent(type="delta", content=after))
        new_buffer = ""

    return events, new_buffer


def flush_buffer(buffer: str) -> list[StreamEvent]:
    """Flush any remaining buffer content as plain text at stream end."""
    if buffer:
        return [StreamEvent(type="delta", content=buffer)]
    return []
