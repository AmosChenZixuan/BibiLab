"""Bounded section derivation — the structural foundation for the section tier.

A `Section` is a contiguous, token-bounded span of the transcript, snapped to
the longest pause inside a cut-zone around a target. The section tier sits
between `sources` and chunks; chunks are then produced *within* a section, so
the chunk → section → source nesting is physical (not by convention).

POC validation: token target + longest-pause won the boundary study on
2026-06-07 (4/5 episode-seam recovery on a 97-min synthetic zh transcript).
Embedding topic-drift and time-target were both rejected. See
docs/specs/2026-06-07-bounded-sections-design.md POC § for evidence.

Tuning: the POC default of 6000 was too aggressive for the live ~20-min zh
video corpus (max sub-20-min source ≈ 6200 tokens, so the 6000 backstop
fired and cut 20-min videos into 2 sections). Tuned to 12000 against
the user's `~/.bibilab/` DB on 2026-06-08: keeps the 20-min corpus
median as 1 section, cuts 30+ min videos. Re-tune if 30+ min videos
stop cutting.

Why the section cap is **flat** (not language-scaled like chunk.py): the
constraint here is the LLM token budget (one section feeds one read_section
/ refine-summarizer call), not chunk readability. Language scaling makes sense
for chunk size; it does not apply here.
"""

import logging
from dataclasses import dataclass, replace

from bibilab.pipeline._shared import count_tokens
from bibilab.pipeline.chunk import RagChunk
from bibilab.pipeline.transcribe import WhisperSegment

logger = logging.getLogger(__name__)

# Tuned for the live ~20-min zh video corpus (2026-06-08). 12000 keeps the
# corpus median as 1 section (max sub-30-min source ≈ 7200 tokens); 30+ min
# videos cut. Re-tune if 30+ min videos stop cutting in production.
SECTION_TARGET_TOKENS = 12000
ZONE_LOW = 0.6
ZONE_HIGH = 1.4


@dataclass
class Section:
    seg_start: int
    seg_end: int
    token_count: int
    timestamp_start: float
    timestamp_end: float

    def __post_init__(self) -> None:
        if self.seg_start < 0:
            raise ValueError(f"seg_start < 0 (got {self.seg_start})")
        if self.seg_end < self.seg_start:
            raise ValueError(f"Invalid seg range: [{self.seg_start}, {self.seg_end}] (seg_end < seg_start)")


def derive_sections(
    segments: list[WhisperSegment],
    language: str,  # noqa: ARG001 — kept for future per-language tuning; flat cap today
    target_tokens: int = SECTION_TARGET_TOKENS,
) -> list[Section]:
    """Walk segments accumulating tokens; in the cut-zone [ZONE_LOW*target,
    ZONE_HIGH*target], cut at the segment boundary with the longest pause
    (i.e., boundary AFTER seg `i` maximizing `segs[i+1].start - segs[i].end`).
    `ZONE_HIGH*target` is a hard backstop if no in-zone pause beats it. The
    trailing remainder is always emitted as the final section.

    Cuts ALWAYS land on a segment boundary — never mid-sentence — so chunks
    produced per-section (Task 4) are guaranteed to nest in exactly one
    section by construction.

    Parameters
    ----------
    segments
        Punctuated sentence segments (output of pipeline/punctuate.py).
    language
        Kept for symmetry with chunk.py and a future per-language tuning; the
        cap is flat today (the LLM token budget doesn't scale with language).
    target_tokens
        Override seam for tests; production always uses SECTION_TARGET_TOKENS.
    """
    n = len(segments)
    if n == 0:
        return []
    seg_tokens = [count_tokens(s.text) for s in segments]

    sections: list[Section] = []
    start = 0
    while start < n:
        best_i: int | None = None
        best_pause = -1.0
        backstop_fired = False
        forced_i = n - 1
        cum = 0
        for i in range(start, n):
            cum += seg_tokens[i]
            if cum < ZONE_LOW * target_tokens:
                continue
            if cum > ZONE_HIGH * target_tokens:
                # Backstop: forced cut lands at the segment that first pushed
                # cumulative tokens past the cap (i). The section ends at
                # that segment; the next section starts at i+1.
                forced_i = i
                backstop_fired = True
                break
            # In zone: consider cutting AFTER seg i (boundary between i and i+1).
            if i < n - 1:
                pause = segments[i + 1].start - segments[i].end
                if pause > best_pause:
                    best_pause = pause
                    best_i = i
        # Priority: a strictly-positive in-zone pause wins (a real seam beats
        # the backstop). Otherwise the backstop wins if it fired. Otherwise
        # fall through to the trailing remainder at n-1.
        if best_i is not None and best_pause > 0:
            cut = best_i
        elif backstop_fired:
            cut = forced_i
        else:
            cut = n - 1
        sec = Section(
            seg_start=start,
            seg_end=cut,
            token_count=sum(seg_tokens[start : cut + 1]),
            timestamp_start=segments[start].start,
            timestamp_end=segments[cut].end,
        )
        sections.append(sec)
        if cut >= n - 1:
            break
        start = cut + 1

    logger.info(
        "derive_sections: %d sections from %d segments (target=%d, zone=[%d, %d])",
        len(sections),
        n,
        target_tokens,
        int(ZONE_LOW * target_tokens),
        int(ZONE_HIGH * target_tokens),
    )
    return sections


def chunk_by_sections(
    segments: list[WhisperSegment],
    sections: list[Section],
    language: str = "en",
) -> list[RagChunk]:
    """Run `chunk_segments` independently inside each section's segment slice,
    re-stamping `sequence_index` and `seg_start`/`seg_end` to source-global
    values so chunks compose cleanly with the rest of the system (citation
    chain, `get_segments_for_ranges`).

    Short video (1 section spanning all) → byte-identical output to calling
    `chunk_segments(segments, language=...)` directly (modulo the dataclass
    `replace`, which is value-preserving).

    Parameters
    ----------
    segments
        The full source-global segments list (same one `derive_sections` saw).
    sections
        The output of `derive_sections(segments, language)`.
    language
        Forwarded to `chunk_segments` for per-section token-target selection
        (zh → 800 tok target, en → 300 tok target, etc.).
    """
    if not sections:
        return []

    from bibilab.pipeline.chunk import chunk_segments  # local import to avoid a cycle

    out: list[RagChunk] = []
    chunk_offset = 0
    for sec in sections:
        slice_ = segments[sec.seg_start : sec.seg_end + 1]
        per_section = chunk_segments(slice_, language=language)
        for c in per_section:
            out.append(
                replace(
                    c,
                    sequence_index=chunk_offset + c.sequence_index,
                    seg_start=sec.seg_start + c.seg_start,
                    seg_end=sec.seg_start + c.seg_end,
                )
            )
        chunk_offset += len(per_section)
    return out
