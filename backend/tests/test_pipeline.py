"""Unit tests for pipeline modules (mocked I/O and LLM)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import bibilab.pipeline.chunk as chunk_module
from bibilab.pipeline._shared import resolve_response_language
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import _SENT_END, RagChunk, chunk_segments
from bibilab.pipeline.transcribe import WhisperSegment

# ---------------------------------------------------------------------------
# audio.py
# ---------------------------------------------------------------------------


def test_extract_audio_success(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    wav = tmp_path / "video.wav"

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        mock_chain = MagicMock()
        mock_ffmpeg.input.return_value = mock_chain
        mock_chain.output.return_value = mock_chain
        mock_chain.overwrite_output.return_value = mock_chain
        mock_chain.run.return_value = (b"", b"")
        mock_ffmpeg.probe.return_value = {"streams": [{"codec_type": "audio"}], "format": {"duration": "10.0"}}
        # Simulate wav being created by ffmpeg
        wav.write_bytes(b"wav")

        result = extract_audio(video)

    assert result == wav
    assert not video.exists()  # source deleted


def test_extract_audio_ffmpeg_error(tmp_path: Path):
    import ffmpeg

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        mock_chain = MagicMock()
        mock_ffmpeg.input.return_value = mock_chain
        mock_chain.output.return_value = mock_chain
        mock_chain.overwrite_output.return_value = mock_chain
        err = ffmpeg.Error("ffmpeg", b"", b"conversion failed")
        mock_ffmpeg.Error = ffmpeg.Error
        mock_ffmpeg.probe.return_value = {"streams": [{"codec_type": "audio"}]}
        mock_chain.run.side_effect = err

        with pytest.raises(PipelineError, match="FFmpeg"):
            extract_audio(video)


def test_extract_audio_raises_on_missing_audio_stream(tmp_path: Path):
    # A video-only file (e.g. TikTok HEVC variant reached via the /best
    # fallback) must fail with a clear message before FFmpeg extraction,
    # not surface the raw FFmpeg "does not contain any stream" dump.
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.probe.return_value = {"streams": [{"codec_type": "video"}]}

        with pytest.raises(PipelineError, match="no audio track"):
            extract_audio(video)

    assert video.exists()  # source kept — nothing was extracted


def test_extract_audio_probe_failure_fails_open_to_ffmpeg_error(tmp_path: Path):
    # An unreadable file must NOT be misreported as audio-less: when the
    # stream probe itself errors, extraction proceeds and FFmpeg's own
    # failure surfaces instead of the no-audio-track message.
    import ffmpeg

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        mock_chain = MagicMock()
        mock_ffmpeg.input.return_value = mock_chain
        mock_chain.output.return_value = mock_chain
        mock_chain.overwrite_output.return_value = mock_chain
        mock_ffmpeg.Error = ffmpeg.Error
        mock_ffmpeg.probe.side_effect = ffmpeg.Error("ffprobe", b"", b"invalid data")
        mock_chain.run.side_effect = ffmpeg.Error("ffmpeg", b"", b"broken file")

        with pytest.raises(PipelineError, match="FFmpeg"):
            extract_audio(video)


def _probe_by_suffix(durations: dict[str, float]):
    """side_effect for ffmpeg.probe: pick a duration by the path's suffix."""

    def _probe(path: str):
        for suffix, dur in durations.items():
            if str(path).endswith(suffix):
                return {"format": {"duration": str(dur)}}
        raise AssertionError(f"unexpected probe path {path}")

    return _probe


def _mock_extract(mock_ffmpeg, wav: Path, durations: dict[str, float]) -> None:
    """Wire the ffmpeg mock to "succeed" and report the given probe durations."""
    import ffmpeg

    mock_chain = MagicMock()
    mock_ffmpeg.input.return_value = mock_chain
    mock_chain.output.return_value = mock_chain
    mock_chain.overwrite_output.return_value = mock_chain
    mock_chain.run.return_value = (b"", b"")
    mock_ffmpeg.Error = ffmpeg.Error
    mock_ffmpeg.probe.side_effect = _probe_by_suffix(durations)
    wav.write_bytes(b"wav")


def test_extract_audio_raises_on_truncated_faststart(tmp_path: Path):
    # Reproduces the bug: container reports full 60s (front moov intact), but
    # ffmpeg decoded only 23.6s — silent truncation. No expected_duration given,
    # so only the container-vs-decoded check fires.
    video = tmp_path / "video.m4a"
    video.write_bytes(b"fake")
    wav = tmp_path / "video.wav"

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        _mock_extract(mock_ffmpeg, wav, {".m4a": 60.0, ".wav": 23.6})
        with pytest.raises(PipelineError, match="audio_truncated"):
            extract_audio(video)

    assert video.exists()  # source NOT deleted on validation failure


def test_extract_audio_raises_below_expected_duration(tmp_path: Path):
    # Container and decoded agree (30s) but the platform's known duration is 60s —
    # the file itself is short. Only the expected-vs-decoded check catches this.
    video = tmp_path / "video.m4a"
    video.write_bytes(b"fake")
    wav = tmp_path / "video.wav"

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        _mock_extract(mock_ffmpeg, wav, {".m4a": 30.0, ".wav": 30.0})
        with pytest.raises(PipelineError, match="audio_truncated"):
            extract_audio(video, expected_duration=60.0)


def test_extract_audio_warns_when_no_reference_available(tmp_path: Path, caplog):
    # Both references unknown: the container can't be probed (0.0) and no
    # expected_duration is given. Nothing can validate the decode, so it must
    # pass — but the unverified skip is logged, never silent.
    video = tmp_path / "video.m4a"
    video.write_bytes(b"fake")
    wav = tmp_path / "video.wav"

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        _mock_extract(mock_ffmpeg, wav, {".m4a": 0.0, ".wav": 20.0})
        with caplog.at_level(logging.WARNING):
            result = extract_audio(video, expected_duration=0.0)

    assert result == wav
    assert not video.exists()
    assert "unverified" in caplog.text


def test_extract_audio_healthy_passes_and_deletes_source(tmp_path: Path):
    # ~0.97 coverage against both signals → passes, source deleted.
    video = tmp_path / "video.m4a"
    video.write_bytes(b"fake")
    wav = tmp_path / "video.wav"

    with patch("bibilab.pipeline.audio.ffmpeg") as mock_ffmpeg:
        _mock_extract(mock_ffmpeg, wav, {".m4a": 60.0, ".wav": 58.0})
        result = extract_audio(video, expected_duration=60.0)

    assert result == wav
    assert not video.exists()


# ---------------------------------------------------------------------------
# chunk.py
# ---------------------------------------------------------------------------


def _seg(text: str, start: float = 0.0, end: float = 1.0) -> WhisperSegment:
    return WhisperSegment(start=start, end=end, text=text)


def test_chunk_empty():
    assert chunk_segments([]) == []


def test_chunk_single_short_segment():
    chunks = chunk_segments([_seg("hello")])
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].sequence_index == 0


def test_chunk_merges_short_segments(monkeypatch):
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 50)
    segs = [_seg(f"word {i}", start=float(i), end=float(i + 1)) for i in range(10)]
    chunks = chunk_segments(segs)
    # All short segments should merge into one or two chunks
    assert len(chunks) < 10
    assert all(isinstance(c, RagChunk) for c in chunks)


def test_chunk_oversized_segment_is_own_chunk():
    # Create a segment that clearly exceeds the production DOC_TOKEN_BUDGET
    big_text = " ".join(["word"] * 500)
    segs = [_seg(big_text)]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0].text == big_text


def test_chunk_sequence_indices_are_consecutive(monkeypatch):
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 20)
    segs = [_seg(f"sentence number {i} in the test", start=float(i), end=float(i + 1)) for i in range(30)]
    chunks = chunk_segments(segs)
    indices = [c.sequence_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_timestamps_correct():
    segs = [
        WhisperSegment(start=10.0, end=20.0, text="first"),
        WhisperSegment(start=20.0, end=30.0, text="second"),
    ]
    chunks = chunk_segments(segs)
    assert chunks[0].timestamp_start == 10.0
    assert chunks[0].timestamp_end == 30.0


def test_chunk_segments_rejects_language_kwarg():
    """Language-dependent sizing is gone entirely — no parameter is left to
    route through, so there's no way to make output vary by language."""
    with pytest.raises(TypeError):
        chunk_segments([_seg("x")], language="en")


# ---------------------------------------------------------------------------
# chunk.py — pause-aware boundary
# ---------------------------------------------------------------------------


def _word_seg(count: int, start: float, end: float) -> WhisperSegment:
    """Segment with `count` tokens of filler text."""
    return _seg("word " * count, start=start, end=end)


def test_chunk_pause_boundary_splits_at_long_gap(monkeypatch):
    """3s gap between groups produces chunk boundary when buffer past min_target."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 100)
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 3s gap (10.0 → 13.0)
    group2 = [
        _word_seg(30, start=13.0, end=18.0),
        _word_seg(30, start=18.0, end=23.0),
    ]
    chunks = chunk_segments(group1 + group2)
    assert len(chunks) == 2
    assert chunks[0].timestamp_end == 10.0
    assert chunks[1].timestamp_start == 13.0


def test_chunk_small_gap_no_pause_split(monkeypatch):
    """Gaps under default threshold fall back to token-ceiling flush only."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 200)
    segs = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.5, end=10.0),  # 0.5s gap
        _word_seg(30, start=10.8, end=15.0),  # 0.8s gap
        _word_seg(30, start=15.0, end=20.0),
    ]
    # All gaps < 1.5s, buffer accumulates to 120 tokens, budget=200 → one chunk
    chunks = chunk_segments(segs)
    assert len(chunks) == 1


def test_chunk_pause_below_min_target_no_split(monkeypatch):
    """Long pause with buffer below min_target_ratio does not trigger a pause flush."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 200)
    segs = [
        _seg("tiny buffer", start=0.0, end=2.0),
        # 3s gap but buffer too small (2 tokens vs min 100 for budget=200)
        _word_seg(30, start=5.0, end=10.0),
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1


def test_chunk_pause_threshold_configurable(monkeypatch):
    """Lower pause_threshold_seconds triggers split on smaller gaps."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 100)
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 1.0s gap (10.0 → 11.0)
    group2 = [_word_seg(30, start=11.0, end=16.0)]
    segs = group1 + group2

    # Default 1.5s threshold: 1.0s gap → no split
    chunks_default = chunk_segments(segs)
    assert len(chunks_default) == 1

    # Custom 0.5s threshold: 1.0s gap → split
    chunks_low = chunk_segments(segs, pause_threshold_seconds=0.5)
    assert len(chunks_low) == 2


def test_chunk_ceiling_forces_flush_even_below_min_target(monkeypatch):
    """The ceiling is a hard invariant — unlike the old target/max split, it
    flushes a tiny buffer rather than let it merge past the ceiling with an
    incoming segment. This is the swallow-path bug #716 fixes: previously a
    buffer under min_target_ratio was allowed to silently absorb an incoming
    segment even when the combined size broke the declared bound.
    """
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 45)
    # buffer = 20 tokens (2 * 10, well under min_flush=22.5), incoming = 30
    # tokens. 20+30=50 > 45 — the ceiling forces a flush of the small buffer
    # instead of absorbing the incoming segment into one 50-token chunk.
    segs = [
        _word_seg(10, start=0.0, end=2.0),
        _word_seg(10, start=2.0, end=4.0),
        _word_seg(30, start=4.0, end=6.0),
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 2
    assert [c.seg_end for c in chunks] == [1, 2]


def test_chunk_pause_flush_before_oversized_segment(monkeypatch):
    """Oversized branch flushes buffer; pause block never reached for it.

    The oversized check runs first in the loop body. When an oversized
    segment arrives, its precursor path flushes any accumulated buffer
    before emitting the oversized segment as its own chunk. This holds
    regardless of pause gaps — the pause-aware block is unreachable
    for oversized segments due to the continue right after it.
    """
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 100)
    # _word_seg(30) = 30 tokens x 2 = 60 tokens buffer (>= 50 min for budget=100)
    # _word_seg(140) = 140 tokens (exceeds the 100-token ceiling) → oversized
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 3s gap (10.0 → 13.0) — oversized path flushes buffer, not pause path
    oversized = _word_seg(140, start=13.0, end=18.0)
    chunks = chunk_segments(group1 + [oversized])
    assert len(chunks) == 2
    assert chunks[0].timestamp_end == 10.0
    assert chunks[1].timestamp_start == 13.0


# ---------------------------------------------------------------------------
# chunk.py — sentence-boundary-aware token flush
# ---------------------------------------------------------------------------


def test_chunk_sentence_end_triggers_flush(caplog, monkeypatch):
    """Segment ending with 。triggers flush at sentence boundary when past the ceiling."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 300)
    filler = "word " * 25  # 25 tokens/seg (word-count)
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler.rstrip() + "。", start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs)

    assert len(chunks) == 2
    assert chunks[0].text.endswith("。")
    # Cut landed on a sentence boundary → no forced-fallback warning.
    assert "non-sentence boundary" not in caplog.text


def test_chunk_no_sentence_end_flushes_at_ceiling(caplog, monkeypatch):
    """Without any sentence-end in buffer, token-flush bounds chunk at the ceiling."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 300)
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(14)]

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs)

    assert len(chunks) == 2
    # No sentence boundary anywhere → token-forced cut warns.
    assert "token-forced=1" in caplog.text
    assert not chunks[0].text.endswith(_SENT_END)


@pytest.mark.parametrize("punct", ["!", "?", "．", "…", "。", "！", "？"])
def test_chunk_punctuation_variants_trigger_sentence_flush(punct, monkeypatch):
    """Each entry in _SENT_END acts as a sentence boundary when scan finds it."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 300)
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler.rstrip() + punct, start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    chunks = chunk_segments(segs)
    assert len(chunks) == 2, f"punct={punct!r} should trigger flush"
    assert chunks[0].text.endswith(punct)


@pytest.mark.parametrize("ambiguous", [".", ";"])
def test_chunk_ascii_period_semicolon_not_sentence_end(ambiguous, caplog, monkeypatch):
    """ASCII '.' and ';' are excluded — decimals, abbreviations, list separators.
    A segment ending in one of these doesn't count as a sentence boundary, so
    the eventual flush is token-forced rather than sentence-recognized (the
    resulting chunk may still incidentally *contain* the ambiguous segment —
    what matters is that it was never treated as a valid cut point)."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 300)
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler.rstrip() + ambiguous, start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs)

    assert len(chunks) == 2
    assert "token-forced=1" in caplog.text


def test_chunk_sentence_boundary_in_middle_of_buffer(caplog, monkeypatch):
    """Sentence boundary at buf[i<-1] still triggers split (scan, not last-only)."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 300)
    filler = "word " * 25
    # s0..s9 = 10 segs no punct (250 tokens). s10 = filler+"。" (25). s11..s14 no punct (100).
    # Incoming s15 (25) → buf = 375 > 300. Scan finds s10 as boundary;
    # head s0..s10 (275 tokens, >= 150) flushes; tail s11..s14 retained.
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(10)]
    segs.append(_seg(filler.rstrip() + "。", start=10.0, end=11.0))
    segs.extend(_seg(filler, start=float(i), end=float(i + 1)) for i in range(11, 16))

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs)

    assert len(chunks) >= 2
    assert chunks[0].text.endswith("。"), "split must land on the boundary, not after it"
    # Sentence-boundary cut → no forced-fallback warning.
    assert "non-sentence boundary" not in caplog.text


def test_chunk_sentence_flush_forces_tiny_chunk_below_min_target(monkeypatch):
    """The ceiling forces a flush even when the sentence-ended buffer is far
    below min_target_ratio — the old target/max headroom that let this merge
    into one oversized chunk is gone; the ceiling wins over chunk-size
    uniformity (same invariant as test_chunk_ceiling_forces_flush_even_below_min_target,
    exercised via the sentence-boundary path instead of the plain forced-cut path).
    """
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 200)
    segs = [
        _seg("hello world this ends。", start=0.0, end=1.0),
        _seg("word " * 220, start=1.0, end=2.0),
    ]

    chunks = chunk_segments(segs)
    assert len(chunks) == 2
    assert chunks[0].text.endswith("。")


# ---------------------------------------------------------------------------
# chunk.py — ceiling invariant (#716)
# ---------------------------------------------------------------------------


def test_chunk_no_emitted_chunk_ever_exceeds_ceiling(monkeypatch):
    """Property: for a swept range of segment sizes and sentence-ending
    placements — exercising the pause, sentence, forced-token, oversized, and
    multi-flush-per-segment paths — no emitted chunk's token count exceeds
    DOC_TOKEN_BUDGET. A deterministic sweep (every segment size 1..budget+5,
    alternating sentence-ended/not), not a random sample, so the check can't
    pass by luck on a few sampled sizes.
    """
    budget = 20
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", budget)
    segs = []
    # Sizes stay under budget — a lone oversized segment is a documented,
    # separate exception (atomic, unsplittable); this sweep targets the
    # merge/split logic's ceiling, not the oversized-segment path.
    for i, size in enumerate(range(1, budget)):
        text = " ".join(f"w{i}_{j}" for j in range(size))
        if size % 2 == 0:
            text += "!"  # sentence-ended for even sizes, plain for odd
        segs.append(_seg(text, start=float(i), end=float(i + 1)))

    chunks = chunk_segments(segs)

    assert len(chunks) > 1  # sanity: the sweep actually forces multiple chunks
    for c in chunks:
        assert chunk_module.count_tokens_xlmr(c.text) <= budget, (
            f"chunk [{c.seg_start}..{c.seg_end}] exceeds the {budget}-token ceiling"
        )


def test_chunk_overshoot_shape_boundary_on_incoming_segment_stays_split(monkeypatch):
    """Regression for the identified bug (issue #716): a short buffered
    segment with no sentence boundary, followed by a segment that alone fits
    under the ceiling but combined with the buffer would exceed it — and
    which itself ends with sentence-ending punctuation. The old code searched
    buf + [incoming] together and accepted a boundary landing on the incoming
    segment with no upper-bound check, flushing buf+seg as one over-ceiling
    chunk. The fix never considers the incoming segment part of the split
    search, so the two must land in separate chunks, each under the ceiling.
    """
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 20)
    small_buffered = _seg("w0 w1 w2 w3 w4", start=0.0, end=1.0)  # 5 tokens, no boundary
    incoming = _seg(
        "w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20!", start=1.0, end=2.0
    )  # 16 tokens, sentence-ended; 5+16=21 > 20

    chunks = chunk_segments([small_buffered, incoming])

    assert len(chunks) == 2, "buffer and incoming segment must not be merged into one over-ceiling chunk"
    for c in chunks:
        assert chunk_module.count_tokens_xlmr(c.text) <= 20
    assert chunks[0].seg_end == 0
    assert chunks[1].seg_start == 1


# ---------------------------------------------------------------------------
# chunk.py — seg-range tracking (P3)
# ---------------------------------------------------------------------------


def test_chunk_seg_range_covers_every_segment_contiguously():
    """3 short sentences, well under the ceiling → one chunk; seg-range spans 0..2."""
    segs = [
        _seg("第一句。", start=0.0, end=1.0),
        _seg("第二句。", start=1.0, end=2.0),
        _seg("第三句。", start=2.0, end=3.0),
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert (chunks[0].seg_start, chunks[0].seg_end) == (0, 2)


def test_chunk_seg_range_partitions_input_with_no_gap_or_overlap(monkeypatch):
    """Force multiple chunks via a tiny ceiling; ranges tile [0, N-1] exactly."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 3)
    segs = [_seg(f"句子{i}。", start=float(i), end=float(i) + 1) for i in range(10)]
    chunks = chunk_segments(segs)
    # contiguous, ascending, no gap, no overlap across the whole input
    assert chunks[0].seg_start == 0
    assert chunks[-1].seg_end == 9
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.seg_start == prev.seg_end + 1


def test_chunk_seg_range_oversized_segment_is_its_own_range(monkeypatch):
    """Oversized segment (index 1) is a standalone chunk with its own range."""
    monkeypatch.setattr(chunk_module, "DOC_TOKEN_BUDGET", 50)
    big = _seg("超长 " * 500, start=0.0, end=5.0)  # exceeds the ceiling → own chunk
    segs = [_seg("短句。", start=5.0, end=6.0), big, _seg("另一句。", start=6.0, end=7.0)]
    chunks = chunk_segments(segs)
    oversized = [c for c in chunks if c.seg_start == 1 and c.seg_end == 1]
    assert len(oversized) == 1


# ---------------------------------------------------------------------------
# _shared.py
# ---------------------------------------------------------------------------


def test_resolve_response_language_with_ui_returns_ui_lang():
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="openai", model="gpt-4o-mini", api_key="k", output_language="ui")
    assert resolve_response_language(cfg, "zh") == "zh"
    assert resolve_response_language(cfg, "en") == "en"


def test_resolve_response_language_with_ui_falls_back_to_en():
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="openai", model="gpt-4o-mini", api_key="k", output_language="ui")
    assert resolve_response_language(cfg, None) == "en"


def test_resolve_response_language_with_explicit_language():
    from bibilab.config import AIConfig

    cfg = AIConfig(protocol="openai", model="gpt-4o-mini", api_key="k", output_language="en")
    assert resolve_response_language(cfg, "zh") == "en"
    cfg_zh = AIConfig(protocol="openai", model="gpt-4o-mini", api_key="k", output_language="zh")
    assert resolve_response_language(cfg_zh, None) == "zh"


# ---------------------------------------------------------------------------
# _build_initial_prompt language instruction
# (replaces the old _generate_artifact tests; the lang-instruction contract
# now lives directly in _build_initial_prompt's call to resolve_response_language,
# _LANG_INSTRUCTION, and _lang_output_directive)
# ---------------------------------------------------------------------------


def test_build_initial_prompt_includes_zh_lang_instruction():
    """_build_initial_prompt prepends 简体中文 language instruction when ui_lang=zh."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.pipeline.artifact_refine import _build_initial_prompt

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="ui",
    )

    prompt = _build_initial_prompt(
        prompt="Generate a summary",
        transcript_text="This is a test transcript.",
        cfg=cfg,
        ui_lang="zh",
    )

    assert prompt.startswith("请用中文回答")
    assert "All output fields MUST be written in 简体中文" in prompt


def test_build_initial_prompt_includes_en_lang_instruction():
    """_build_initial_prompt prepends English language instruction when output_language=en."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.pipeline.artifact_refine import _build_initial_prompt

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="en",
    )

    prompt = _build_initial_prompt(
        prompt="Generate a summary",
        transcript_text="This is a test transcript.",
        cfg=cfg,
        ui_lang=None,
    )

    assert prompt.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in prompt


def test_build_initial_prompt_unknown_lang_falls_back_to_english():
    """_build_initial_prompt with unrecognized output_language falls back to English."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.pipeline.artifact_refine import _build_initial_prompt

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="fr",
    )

    prompt = _build_initial_prompt(
        prompt="Generate a summary",
        transcript_text="French biased transcript",
        cfg=cfg,
        ui_lang=None,
    )

    assert prompt.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in prompt
