"""Unit tests for pipeline modules (mocked I/O and LLM)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        mock_chain.run.side_effect = err

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
# transcribe.py
# ---------------------------------------------------------------------------


def test_transcribe_dispatches_large_v3_to_funasr(tmp_path: Path):
    from bibilab.config import TranscriptionConfig
    from bibilab.pipeline import transcribe as t_mod

    cfg = TranscriptionConfig(model="large-v3", device="cpu", language="auto")
    funasr_out = ([WhisperSegment(start=0.0, end=1.0, text="hi")], "en")
    with patch.object(t_mod, "_transcribe_funasr", return_value=funasr_out) as mock_f:
        result = t_mod.transcribe(tmp_path / "a.wav", cfg)
    mock_f.assert_called_once()
    assert result == funasr_out


def test_transcribe_dispatches_to_sensevoice(tmp_path: Path):
    from bibilab.config import TranscriptionConfig
    from bibilab.pipeline import transcribe as t_mod

    cfg = TranscriptionConfig(model="sensevoice-small", device="cpu", language="auto")
    expected = ([WhisperSegment(start=0.0, end=1.0, text="ni")], None)
    with patch.object(t_mod, "_transcribe_funasr", return_value=expected) as mock_sv:
        result = t_mod.transcribe(tmp_path / "a.wav", cfg)
    mock_sv.assert_called_once()
    assert result == expected


def test_transcribe_unknown_model_raises(tmp_path: Path):
    from bibilab.config import TranscriptionConfig
    from bibilab.pipeline import transcribe as t_mod
    from bibilab.pipeline.audio import PipelineError

    cfg = TranscriptionConfig(model="parakeet", device="cpu", language="auto")
    with pytest.raises(PipelineError, match="Unknown model"):
        t_mod.transcribe(tmp_path / "a.wav", cfg)


def test_transcribe_funasr_propagates_speaker_label(tmp_path: Path):
    """When CAM++ is downloaded, FunASR returns spk per sentence_info; transcribe maps to speaker."""
    from bibilab.config import TranscriptionConfig
    from bibilab.pipeline import transcribe as t_mod

    cfg = TranscriptionConfig(model="sensevoice-small", device="cpu", language="auto")

    class FakeFunasr:
        def generate(self, **kwargs):
            return [
                {
                    "sentence_info": [
                        {"text": "hello", "start": 0.0, "end": 1.0, "spk": 0},
                        {"text": "world", "start": 1.0, "end": 2.0, "spk": 1},
                    ],
                    "language": "en",
                }
            ]

    with patch.object(t_mod, "_load_funasr", return_value=FakeFunasr()):
        segs, lang = t_mod._transcribe_funasr(tmp_path / "a.wav", cfg)

    assert [(s.text, s.speaker) for s in segs] == [("hello", "SPK_0"), ("world", "SPK_1")]
    assert lang == "en"


def test_transcribe_funasr_missing_spk_field_leaves_speaker_none(tmp_path: Path):
    from bibilab.config import TranscriptionConfig
    from bibilab.pipeline import transcribe as t_mod

    cfg = TranscriptionConfig(model="sensevoice-small", device="cpu", language="auto")

    class FakeFunasr:
        def generate(self, **kwargs):
            return [{"sentence_info": [{"text": "lone", "start": 0.0, "end": 1.0}]}]

    with patch.object(t_mod, "_load_funasr", return_value=FakeFunasr()):
        segs, _ = t_mod._transcribe_funasr(tmp_path / "a.wav", cfg)

    assert segs[0].speaker is None


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


def test_chunk_merges_short_segments():
    segs = [_seg(f"word {i}", start=float(i), end=float(i + 1)) for i in range(10)]
    chunks = chunk_segments(segs, target_tokens=50)
    # All short segments should merge into one or two chunks
    assert len(chunks) < 10
    assert all(isinstance(c, RagChunk) for c in chunks)


def test_chunk_oversized_segment_is_own_chunk():
    # Create a segment that clearly exceeds MAX_TOKENS (400)
    big_text = " ".join(["word"] * 500)
    segs = [_seg(big_text)]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0].text == big_text


def test_chunk_sequence_indices_are_consecutive():
    segs = [_seg(f"sentence number {i} in the test", start=float(i), end=float(i + 1)) for i in range(30)]
    chunks = chunk_segments(segs, target_tokens=20)
    indices = [c.sequence_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_timestamps_correct():
    segs = [
        WhisperSegment(start=10.0, end=20.0, text="first"),
        WhisperSegment(start=20.0, end=30.0, text="second"),
    ]
    chunks = chunk_segments(segs, target_tokens=1000)
    assert chunks[0].timestamp_start == 10.0
    assert chunks[0].timestamp_end == 30.0


def test_chunk_language_zh_reduces_fragmentation():
    """Chinese gets higher target_tokens, producing fewer chunks than English."""
    seg = _seg("this sentence has approximately twenty tokens in cl100k base encoding")
    segs = [seg for _ in range(50)]
    chunks_en = chunk_segments(segs, language="en")
    chunks_zh = chunk_segments(segs, language="zh")
    assert len(chunks_zh) < len(chunks_en)


def test_chunk_explicit_target_overrides_language():
    """Explicit target_tokens bypasses the language lookup table."""
    seg = _seg("this sentence has approximately twenty tokens in cl100k base encoding")
    segs = [seg for _ in range(50)]
    chunks_a = chunk_segments(segs, target_tokens=500, language="en")
    chunks_b = chunk_segments(segs, target_tokens=500, language="zh")
    assert len(chunks_a) == len(chunks_b)


def test_chunk_unknown_language_falls_back_to_default():
    """Unrecognized language code uses _DEFAULT_TARGET_TOKENS (300)."""
    seg = _seg("this sentence has approximately twenty tokens in cl100k base encoding")
    segs = [seg for _ in range(50)]
    chunks_fr = chunk_segments(segs, language="fr")
    chunks_en = chunk_segments(segs, language="en")
    # French is unknown, falls back to default (same as English)
    assert len(chunks_fr) == len(chunks_en)


# ---------------------------------------------------------------------------
# chunk.py — pause-aware boundary
# ---------------------------------------------------------------------------


def _word_seg(count: int, start: float, end: float) -> WhisperSegment:
    """Segment with `count` tokens of filler text."""
    return _seg("word " * count, start=start, end=end)


def test_chunk_pause_boundary_splits_at_long_gap():
    """3s gap between groups produces chunk boundary when buffer past min_target."""
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 3s gap (10.0 → 13.0)
    group2 = [
        _word_seg(30, start=13.0, end=18.0),
        _word_seg(30, start=18.0, end=23.0),
    ]
    chunks = chunk_segments(group1 + group2, target_tokens=100)
    assert len(chunks) == 2
    assert chunks[0].timestamp_end == 10.0
    assert chunks[1].timestamp_start == 13.0


def test_chunk_small_gap_no_pause_split():
    """Gaps under default threshold fall back to token-target flush only."""
    segs = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.5, end=10.0),  # 0.5s gap
        _word_seg(30, start=10.8, end=15.0),  # 0.8s gap
        _word_seg(30, start=15.0, end=20.0),
    ]
    # All gaps < 1.5s, buffer accumulates to ~120 tokens, target=200 → one chunk
    chunks = chunk_segments(segs, target_tokens=200)
    assert len(chunks) == 1


def test_chunk_pause_below_min_target_no_split():
    """Long pause with buffer below min_target_ratio does not trigger flush."""
    segs = [
        _seg("tiny buffer", start=0.0, end=2.0),
        # 3s gap but buffer too small (~2 tokens vs min 100 for target=200)
        _word_seg(30, start=5.0, end=10.0),
    ]
    chunks = chunk_segments(segs, target_tokens=200)
    assert len(chunks) == 1


def test_chunk_pause_threshold_configurable():
    """Lower pause_threshold_seconds triggers split on smaller gaps."""
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 1.0s gap (10.0 → 11.0)
    group2 = [_word_seg(30, start=11.0, end=16.0)]
    segs = group1 + group2

    # Default 1.5s threshold: 1.0s gap → no split
    chunks_default = chunk_segments(segs, target_tokens=100)
    assert len(chunks_default) == 1

    # Custom 0.5s threshold: 1.0s gap → split
    chunks_low = chunk_segments(segs, target_tokens=100, pause_threshold_seconds=0.5)
    assert len(chunks_low) == 2


def test_chunk_token_flush_skipped_when_buffer_below_min_target():
    """Token-target flush skips when buffer < min_target_ratio to avoid orphans.

    After a pause flush empties the buffer, a single small segment may sit
    alone. When the next segment would overflow, the min_target guard skips
    the flush, letting the small buffer merge into a larger chunk instead.
    """
    # _word_seg(10) ≈ 11 tokens, _word_seg(30) ≈ 31 tokens
    # target=50, min=25 (50*0.5), max=66 (50*4/3)
    segs = [
        _word_seg(10, start=0.0, end=2.0),
        _word_seg(10, start=2.0, end=4.0),  # buffer ~22t (< 25 min)
        _word_seg(30, start=4.0, end=6.0),  # 22+31=53 > 50, but 22<25 → skip
    ]
    chunks = chunk_segments(segs, target_tokens=50)
    assert len(chunks) == 1


def test_chunk_pause_flush_before_oversized_segment():
    """Oversized branch flushes buffer; pause block never reached for it.

    The oversized check runs first in the loop body. When an oversized
    segment arrives, its precursor path flushes any accumulated buffer
    before emitting the oversized segment as its own chunk. This holds
    regardless of pause gaps — the pause-aware block is unreachable
    for oversized segments due to the continue on line 89.
    """
    # _word_seg(30) ≈ 31 tokens × 2 = ~62 tokens buffer (>= 50 min for target=100)
    # _word_seg(140) ≈ 141 tokens (> 133 max) → oversized
    group1 = [
        _word_seg(30, start=0.0, end=5.0),
        _word_seg(30, start=5.0, end=10.0),
    ]
    # 3s gap (10.0 → 13.0) — oversized path flushes buffer, not pause path
    oversized = _word_seg(140, start=13.0, end=18.0)
    chunks = chunk_segments(group1 + [oversized], target_tokens=100)
    assert len(chunks) == 2
    assert chunks[0].timestamp_end == 10.0
    assert chunks[1].timestamp_start == 13.0


# ---------------------------------------------------------------------------
# chunk.py — sentence-boundary-aware token flush
# ---------------------------------------------------------------------------


def test_chunk_sentence_end_triggers_flush(caplog):
    """Segment ending with 。triggers flush at sentence boundary when past target."""
    filler = "word " * 25  # 26 tokens/seg with cl100k
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler + "。", start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs, target_tokens=300)

    assert len(chunks) == 2
    assert chunks[0].text.endswith("。")
    # Cut landed on a sentence boundary → no forced-fallback warning.
    assert "non-sentence boundary" not in caplog.text


def test_chunk_no_sentence_end_flushes_at_target(caplog):
    """Without any sentence-end in buffer, token-flush bounds chunk at target."""
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(14)]

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs, target_tokens=300)

    assert len(chunks) == 2
    # No sentence boundary anywhere → token-forced cut warns.
    assert "token-forced=1" in caplog.text
    assert not chunks[0].text.endswith(_SENT_END)


@pytest.mark.parametrize("punct", ["!", "?", "．", "…", "。", "！", "？"])
def test_chunk_punctuation_variants_trigger_sentence_flush(punct):
    """Each entry in _SENT_END acts as a sentence boundary when scan finds it."""
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler + punct, start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    chunks = chunk_segments(segs, target_tokens=300)
    assert len(chunks) == 2, f"punct={punct!r} should trigger flush"
    assert chunks[0].text.endswith(punct)


@pytest.mark.parametrize("ambiguous", [".", ";"])
def test_chunk_ascii_period_semicolon_not_sentence_end(ambiguous):
    """ASCII '.' and ';' are excluded — decimals, abbreviations, list separators."""
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(11)]
    segs.append(_seg(filler + ambiguous, start=11.0, end=12.0))
    segs.append(_seg(filler, start=12.0, end=13.0))

    chunks = chunk_segments(segs, target_tokens=300)
    # Buffer flushes at target (token branch), not on the ambiguous character.
    assert len(chunks) == 2
    assert not chunks[0].text.endswith(ambiguous)


def test_chunk_sentence_boundary_in_middle_of_buffer(caplog):
    """Sentence boundary at buf[i<-1] still triggers split (scan, not last-only)."""
    filler = "word " * 25
    # s0..s9 = 10 segs no punct (260 tokens). s10 = filler+"。" (26). s11..s14 no punct (104).
    # Incoming s15 (26) → buf = 390 > 300. Scan finds s10 as boundary;
    # head s0..s10 (286 tokens, >= 150) flushes; tail s11..s14 retained.
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(10)]
    segs.append(_seg(filler + "。", start=10.0, end=11.0))
    segs.extend(_seg(filler, start=float(i), end=float(i + 1)) for i in range(11, 16))

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs, target_tokens=300)

    assert len(chunks) >= 2
    assert chunks[0].text.endswith("。"), "split must land on the boundary, not after it"
    # Sentence-boundary cut → no forced-fallback warning.
    assert "non-sentence boundary" not in caplog.text


def test_chunk_sentence_flush_below_min_target_skips():
    """Buffer ends at 。but is below min_target_ratio → boundary not qualifying, segment merged in."""
    segs = [
        _seg("hello world this ends。", start=0.0, end=1.0),
        _seg("word " * 220, start=1.0, end=2.0),
    ]

    chunks = chunk_segments(segs, target_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text.endswith(("word", "word "))


# ---------------------------------------------------------------------------
# chunk.py — seg-range tracking (P3)
# ---------------------------------------------------------------------------


def test_chunk_seg_range_covers_every_segment_contiguously():
    """3 short zh sentences → one chunk; seg-range spans 0..2."""
    segs = [
        _seg("第一句。", start=0.0, end=1.0),
        _seg("第二句。", start=1.0, end=2.0),
        _seg("第三句。", start=2.0, end=3.0),
    ]
    chunks = chunk_segments(segs, target_tokens=1000, language="zh")
    assert len(chunks) == 1
    assert (chunks[0].seg_start, chunks[0].seg_end) == (0, 2)


def test_chunk_seg_range_partitions_input_with_no_gap_or_overlap():
    """Force multiple chunks via tiny token target; ranges tile [0, N-1] exactly."""
    segs = [_seg(f"句子{i}。", start=float(i), end=float(i) + 1) for i in range(10)]
    chunks = chunk_segments(segs, target_tokens=2, chunk_max_tokens=3, language="zh")
    # contiguous, ascending, no gap, no overlap across the whole input
    assert chunks[0].seg_start == 0
    assert chunks[-1].seg_end == 9
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.seg_start == prev.seg_end + 1


def test_chunk_seg_range_oversized_segment_is_its_own_range():
    """Oversized segment (index 1) is a standalone chunk with its own range."""
    big = _seg("超长" * 500, start=0.0, end=5.0)  # exceeds max → own chunk
    segs = [_seg("短句。", start=5.0, end=6.0), big, _seg("另一句。", start=6.0, end=7.0)]
    chunks = chunk_segments(segs, target_tokens=50, language="zh")
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
    from bibilab.worker import _build_initial_prompt

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
    from bibilab.worker import _build_initial_prompt

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
    from bibilab.worker import _build_initial_prompt

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
