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
)


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
