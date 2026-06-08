"""Pure-function tests for section derivation + chunk-by-section re-stamping.

No DB. The pipeline reorder (worker) and the persistence writers (db.py) are
tested elsewhere; here we cover the algorithmic core that the worker calls.
"""

import pytest

from bibilab.pipeline.section import (
    SECTION_TARGET_TOKENS,
    ZONE_HIGH,
    ZONE_LOW,
    Section,
    derive_sections,
)
from bibilab.pipeline.transcribe import WhisperSegment


def test_constants_match_poc_validated_values():
    # Tuned for the live ~20-min zh video corpus (2026-06-08). See the
    # module docstring in `section.py` for the data-driven tuning rationale.
    assert SECTION_TARGET_TOKENS == 12000
    assert ZONE_LOW == 0.6
    assert ZONE_HIGH == 1.4


def test_section_dataclass_accepts_valid_range():
    sec = Section(seg_start=10, seg_end=20, token_count=5800, timestamp_start=100.0, timestamp_end=200.0)
    assert sec.seg_start == 10
    assert sec.seg_end == 20
    assert sec.token_count == 5800


def test_section_dataclass_rejects_invalid_range():
    with pytest.raises(ValueError, match="seg_start < 0"):
        Section(seg_start=-1, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0)


def test_section_dataclass_rejects_end_before_start():
    with pytest.raises(ValueError, match="seg_end < seg_start"):
        Section(seg_start=10, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0)


def test_section_dataclass_allows_equal_end_and_start():
    # A 1-segment section has seg_start == seg_end. The constructor must allow it.
    sec = Section(seg_start=0, seg_end=0, token_count=42, timestamp_start=0.0, timestamp_end=1.0)
    assert sec.seg_start == sec.seg_end == 0


def _seg(start: float, end: float, text: str, speaker: str = "S") -> WhisperSegment:
    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


def count_tokens_for_test(text: str) -> int:
    # Local import to keep test imports tight
    from bibilab.pipeline._shared import count_tokens

    return count_tokens(text)


def test_derive_sections_empty_input():
    assert derive_sections([], "en") == []


def test_derive_sections_single_segment_yields_one_section():
    segs = [_seg(0.0, 1.0, "hello world.")]
    secs = derive_sections(segs, "en")
    assert len(secs) == 1
    assert secs[0].seg_start == 0
    assert secs[0].seg_end == 0
    assert secs[0].token_count == count_tokens_for_test("hello world.")
    assert secs[0].timestamp_start == 0.0
    assert secs[0].timestamp_end == 1.0


def test_derive_sections_short_video_yields_one_section_spanning_all():
    # 50 short segments, well below SECTION_TARGET_TOKENS — must be 1 section
    # covering all of them (the regression guard from the issue's acceptance
    # criteria: "Short video produces exactly 1 section row").
    segs = [_seg(float(i), float(i + 1), f"short sentence {i}.") for i in range(50)]
    secs = derive_sections(segs, "en")
    assert len(secs) == 1
    assert secs[0].seg_start == 0
    assert secs[0].seg_end == 49


def test_derive_sections_long_video_respects_cap():
    # Build ~18K tokens worth of segments to force ≥2 sections. The cap is
    # ZONE_HIGH * SECTION_TARGET_TOKENS = 1.4 * 12000 = 16800 tokens. We add
    # slack for the backstop segment: when cum crosses ZONE_HIGH*target, the
    # segment that pushed us over IS part of the section, so a section can be
    # up to ZONE_HIGH*target + 1 segment (~200 tokens) over the cap.
    n = 180
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(n)]
    secs = derive_sections(segs, "en")
    assert len(secs) >= 2
    for s in secs:
        assert s.token_count <= int(ZONE_HIGH * SECTION_TARGET_TOKENS) + 200, (
            f"section token_count={s.token_count} exceeds cap {ZONE_HIGH * SECTION_TARGET_TOKENS + 200}"
        )


def test_derive_sections_sections_are_contiguous_no_gap_no_overlap():
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(180)]
    secs = derive_sections(segs, "en")
    # Every internal boundary is contiguous: sec[k+1].seg_start == sec[k].seg_end + 1
    for prev, nxt in zip(secs, secs[1:]):
        assert nxt.seg_start == prev.seg_end + 1, (
            f"gap or overlap between sec[{prev.seg_start}..{prev.seg_end}] and sec[{nxt.seg_start}..{nxt.seg_end}]"
        )
    # First section starts at 0, last section ends at len(segs)-1
    assert secs[0].seg_start == 0
    assert secs[-1].seg_end == len(segs) - 1


def test_derive_sections_cuts_land_on_segment_boundaries():
    # Cuts must NEVER split a segment — every section's seg_start/seg_end is
    # a valid segment index in the input list.
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(180)]
    secs = derive_sections(segs, "en")
    for s in secs:
        assert 0 <= s.seg_start <= s.seg_end < len(segs)


def test_derive_sections_picks_longest_pause_in_zone():
    # 3 segments; section cap is small; one segment has a long pause after it.
    # The cut should land at the end of the long-pause segment.
    segs = [
        _seg(0.0, 1.0, ("word " * 100).strip() + "."),  # 0
        _seg(1.0, 2.0, ("word " * 100).strip() + "."),  # 1 — short gap
        _seg(5.0, 6.0, ("word " * 100).strip() + "."),  # 2 — LONG pause (3s)
        _seg(6.0, 7.0, ("word " * 100).strip() + "."),  # 3
    ]
    secs = derive_sections(segs, "en", target_tokens=200)
    # With target=200 and 100-tok segments, we'd cut around segment 1-2.
    # The longest pause is between seg 1 and seg 2 (gap = 3.0s). The cut
    # should be AFTER seg 1, so the first section is [0..1].
    assert secs[0].seg_end == 1, f"expected first cut after seg 1 (longest pause), got {secs[0].seg_end}"


def test_derive_sections_zone_high_backstop_when_no_in_zone_pause():
    # All pauses tiny; cumulative tokens grow past ZONE_HIGH*target without
    # an in-zone cut. Algorithm must cut at ZONE_HIGH*target (forced).
    # Use target=200, ZONE_HIGH=1.4 → backstop at 280 tokens.
    # Build 10 segments of 100 tokens each, no pause.
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(10)]
    # Override target via the (private) parameter — production code passes
    # SECTION_TARGET_TOKENS, but the algorithm is parameterized for testability.
    secs = derive_sections(segs, "en", target_tokens=200)
    # First cut is forced: at the first segment that pushes cumulative past 280.
    # That happens at the 3rd segment (3 * 100 = 300 > 280). Cut after seg 2 → [0..2].
    assert secs[0].seg_end == 2


def test_derive_sections_trailing_remainder_is_emitted():
    # The final section is always emitted even if shorter than ZONE_LOW*target.
    segs = [_seg(float(i), float(i + 1), "hi.") for i in range(5)]  # 5 tiny segments
    secs = derive_sections(segs, "en", target_tokens=10_000)
    # 1 section covering all 5 segments (cumulative is well below ZONE_LOW*target
    # so no in-zone cut fires; final cut is the remainder).
    assert len(secs) == 1
    assert secs[0].seg_start == 0
    assert secs[0].seg_end == 4
