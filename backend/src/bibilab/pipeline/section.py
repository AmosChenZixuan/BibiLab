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
from dataclasses import dataclass

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
