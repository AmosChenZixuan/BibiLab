"""Tests for bibilab.pipeline.punctuate — char-offset alignment is model-free."""

import pytest

from bibilab.pipeline.chunk import _SENT_END
from bibilab.pipeline.punctuate import _align, _strip_punc
from bibilab.pipeline.transcribe import WhisperSegment


def _seg(text, start, end, speaker="SPK_0"):
    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


def test_strip_punc_removes_punctuation_and_spaces():
    assert _strip_punc("天花。板，明 显") == "天花板明显"


def test_align_invariant_violation_raises():
    # ct-punc must only INSERT; a rewritten char breaks the mapping.
    segs = [_seg("天花板", 0.0, 5.0)]
    with pytest.raises(ValueError):
        _align(segs, "天花版。")  # 板 -> 版 is a rewrite


def test_align_basic_single_speaker_two_sentences():
    # one VAD seg, ct-punc inserts a sentence break -> two sentences (G2: share span)
    segs = [_seg("天花板明显是地板", 10.0, 25.0)]
    out = _align(segs, "天花板。明显是地板。")
    assert [s.text for s in out] == ["天花板。", "明显是地板。"]
    # G2: both inherit the single VAD seg's span
    assert all(s.start == 10.0 and s.end == 25.0 for s in out)
    assert all(s.speaker == "SPK_0" for s in out)


def test_align_g1_sentence_spans_two_vad_segments():
    # ct-punc heals a false split across two VAD segs -> one sentence
    # G1: start = first seg.start, end = last seg.end
    segs = [_seg("天花", 10.0, 12.0), _seg("板明显", 12.0, 16.0)]
    out = _align(segs, "天花板明显。")
    assert [s.text for s in out] == ["天花板明显。"]
    assert out[0].start == 10.0 and out[0].end == 16.0


def test_align_g3_speaker_change_forces_cut_without_punctuation():
    # speaker switches mid-stream with no terminal punctuation -> two segments, first has no terminal
    segs = [_seg("你好", 0.0, 2.0, "SPK_0"), _seg("再见", 2.0, 4.0, "SPK_1")]
    out = _align(segs, "你好再见。")
    assert [(s.text, s.speaker) for s in out] == [("你好", "SPK_0"), ("再见。", "SPK_1")]
    assert not out[0].text.rstrip().endswith(_SENT_END)


def test_align_phantom_punctuation_only_segment_dissolves():
    # a VAD seg that is all punctuation contributes no chars -> not its own sentence
    segs = [_seg("你好。", 0.0, 2.0), _seg("。", 2.0, 3.0), _seg("再见", 3.0, 5.0)]
    out = _align(segs, "你好。再见。")
    assert [s.text for s in out] == ["你好。", "再见。"]


def test_align_trailing_buffer_flushed_without_terminal_punctuation():
    segs = [_seg("没有句号", 0.0, 3.0)]
    out = _align(segs, "没有句号")  # ct-punc left no terminal punctuation
    assert [s.text for s in out] == ["没有句号"]
    assert out[0].start == 0.0 and out[0].end == 3.0
