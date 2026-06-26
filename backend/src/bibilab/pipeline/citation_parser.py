"""Incremental citation parser — strips citation tokens from LLM deltas, emits citation events.

Recognized wrappers (D1, case-insensitive on ``Source``), one optional space
between label and digits, closing bracket must match the opening family:
``[N]`` ``[Source N]`` ``(Source N)`` ``（Source N）`` ``[来源N]`` ``（来源N）``
``(来源N)`` ``【N】`` ``【来源N】`` ``［N］`` ``［Source N］`` ``［来源N］``.
Inside one wrapper, digit groups separated by ``,`` ``，`` ``、`` (optional
spaces) emit one citation event per index (D2). An optional ``@M:SS`` clock
timestamp after the index list is silently consumed — the fence header and
turn lines render time as ``@M:SS`` (or ``@H:MM:SS``), so the LLM copies that
shape; a range joins two clocks with ``-``, ``–`` or ``—``.
"""

import json
import logging
import re

from bibilab.pipeline._shared import StreamEvent
from bibilab.pipeline.chat_tools import CitationRegistryEntry

logger = logging.getLogger(__name__)

# Opening bracket → its matching closing bracket (the wrapper families).
# Fullwidth square ［］ is included because the LLM sometimes emits it; fullwidth
# parens （） are the existing symmetric case (added in #323).
_BRACKET_PAIRS = {"[": "]", "(": ")", "（": "）", "【": "】", "［": "］"}
_OPENERS = "".join(_BRACKET_PAIRS)

# Multi-index separators (D2): ASCII / fullwidth comma, ideographic comma.
_SEP = r"[,，、]"
_OPEN_CLASS = r"[\[\(（【［]"

# Clock timestamp the LLM may append (matching the @M:SS / @H:MM:SS shape the
# fence renders); a range joins two clocks with -, – or —. Silently consumed.
_CLOCK = r"\d+:\d{2}(?::\d{2})?"
_TS_SUFFIX = rf"(?: *@ *{_CLOCK}(?: *[-–—] *{_CLOCK})?)?"

# Inner shape: optional label (Source / 来源) + optional single space, then
# digit groups separated by _SEP with optional surrounding spaces (D2).
_INNER = rf"(?:[Ss][Oo][Uu][Rr][Cc][Ee]|来源)? ?(\d+(?: *{_SEP} *\d+)*){_TS_SUFFIX}"

# Full wrapper regex: capture group 1 = opener, group 2 = inner index list,
# group 3 = closer. Closer validated against the opener family in parse_delta.
_CITATION_RE = re.compile(rf"({_OPEN_CLASS})" + _INNER + r"([\]\)）】］])")

# A trailing fragment is held for the next delta iff it is a viable prefix of a
# token: an opener, optionally followed by a partial label, a partial index
# list, and/or a partial @M:SS suffix that has not yet hit its closing bracket (D5).
_VIABLE_PREFIX_RE = re.compile(
    rf"{_OPEN_CLASS}(?:[Ss]?[Oo]?[Uu]?[Rr]?[Cc]?[Ee]?|来?源?) ?\d*(?: *{_SEP} *\d*)*(?: *@[\d :–—-]*)?$"
)


def _expand_indices(
    index_list: str,
    matched: str,
    index_to_entry: dict[int, CitationRegistryEntry],
) -> list[StreamEvent]:
    """Expand one wrapper's index list into per-index citation / text events (D2/D3)."""
    events: list[StreamEvent] = []
    tokens = re.split(_SEP, index_list)
    for token in tokens:
        n = int(token.strip())
        entry = index_to_entry.get(n)
        if entry is not None and entry.citable:
            # chunk_ids snapshot is safe: LLM only emits prose after the
            # retrieve tool result completes — registry is stable at this point.
            # If retrieve ever streams mid-prose, this becomes a race.
            events.append(
                StreamEvent(
                    type="citation",
                    content=json.dumps(
                        {
                            "index": n,
                            "section_id": entry.section_id,
                            "source_id": entry.source_id,
                            "timestamp_start": entry.timestamp_start or 0.0,
                            "chunk_ids": sorted(entry.chunk_ids),
                        }
                    ),
                )
            )
        else:
            # Non-citable entry or unknown index — emit as text.
            if entry is None:
                logger.warning(
                    "citation_hallucinated_index index=%d registry_size=%d",
                    n,
                    len(index_to_entry),
                )
            # D3: a single unknown index emits its original substring; inside a
            # multi-index token each unknown index emits its own digits as text.
            text = matched if len(tokens) == 1 else str(n)
            events.append(StreamEvent(type="delta", content=text))
    return events


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
        if _BRACKET_PAIRS[m.group(1)] != m.group(3):
            continue  # closing bracket must match the opening family
        start, end = m.span()

        if start > pos:
            events.append(StreamEvent(type="delta", content=buffer[pos:start]))

        events.extend(_expand_indices(m.group(2), m.group(0), index_to_entry))
        pos = end

    # Hold a trailing partial token for the next delta (D5): only when the tail
    # from the last opener is a viable prefix of a citation token.
    after = buffer[pos:]
    last_open = max((after.rfind(o) for o in _OPENERS), default=-1)
    if last_open != -1 and _VIABLE_PREFIX_RE.match(after[last_open:]):
        if last_open > 0:
            events.append(StreamEvent(type="delta", content=after[:last_open]))
        new_buffer = after[last_open:]
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
