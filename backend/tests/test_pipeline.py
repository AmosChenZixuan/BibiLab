"""Unit tests for pipeline modules (mocked I/O and LLM)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.config import TranscriptionConfig
from bibilab.pipeline import transcribe as transcribe_mod
from bibilab.pipeline._shared import _resolved_lang
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import _SENT_END, RagChunk, chunk_segments
from bibilab.pipeline.extract import generate_overview
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    _compute_type_for_device,
    transcribe,
    write_transcript,
)

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


# ---------------------------------------------------------------------------
# transcribe.py
# ---------------------------------------------------------------------------


def test_write_transcript(tmp_path: Path):
    segs = [
        WhisperSegment(start=0.0, end=5.0, text="Hello world"),
        WhisperSegment(start=5.0, end=10.0, text="Second segment"),
    ]
    with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
        (tmp_path / "transcripts").mkdir()
        path = write_transcript(segs, "BV1abc")

    lines = path.read_text().splitlines()
    assert lines[0] == "[00:00:00] Hello world"
    assert lines[1] == "[00:00:05] Second segment"


def test_write_transcript_hours(tmp_path: Path):
    segs = [WhisperSegment(start=3661.0, end=3665.0, text="Late segment")]
    with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
        (tmp_path / "transcripts").mkdir()
        path = write_transcript(segs, "BV2xyz")

    assert path.read_text().strip() == "[01:01:01] Late segment"


def test_compute_type_for_device():
    assert _compute_type_for_device("cpu") == "int8"
    assert _compute_type_for_device("cuda") == "float16"


def _fake_whisper_pipeline(monkeypatch, segs=None) -> MagicMock:
    """Replace _load_whisper with a MagicMock whose .transcribe returns (segs, info)."""
    # Disable diarization in tests
    monkeypatch.setattr(transcribe_mod, "is_diarization_model_downloaded", lambda: False)

    model = MagicMock()
    info = MagicMock()
    info.language = "zh"
    fake_segs = []
    for entry in segs or []:
        if len(entry) == 4:
            start, end, text, no_speech = entry
        else:
            start, end, text = entry
            no_speech = 0.05
        m = MagicMock()
        m.start, m.end, m.text, m.no_speech_prob = start, end, text, no_speech
        fake_segs.append(m)
    model.transcribe.return_value = (fake_segs, info)
    monkeypatch.setattr(transcribe_mod, "_load_whisper", lambda cfg: model)
    return model


@pytest.mark.parametrize(
    "lang,expected_strategy_key",
    [
        # 'auto' uses default strategy — applying a per-language prompt before
        # language detection would bias decoding toward that language's tokens.
        ("auto", None),
        ("zh", "zh"),
        ("en", None),
        ("ja", None),
    ],
)
def test_transcribe_language_dispatch(monkeypatch, tmp_path: Path, lang, expected_strategy_key):
    model = _fake_whisper_pipeline(monkeypatch)
    cfg = TranscriptionConfig(language=lang)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    transcribe(audio, cfg)

    expected = (
        transcribe_mod._LANG_STRATEGIES[expected_strategy_key]
        if expected_strategy_key
        else transcribe_mod._DEFAULT_STRATEGY
    )
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["initial_prompt"] == expected.initial_prompt
    assert kwargs["hotwords"] == expected.hotwords
    assert kwargs["condition_on_previous_text"] is expected.condition_on_previous_text


def test_transcribe_strips_silent_prompt_echo(monkeypatch, tmp_path: Path):
    """Prompt-shaped segments on silent windows are dropped."""
    prompt = "以下是普通话的句子，请使用标点符号。"
    _fake_whisper_pipeline(
        monkeypatch,
        segs=[
            (0.0, 2.0, "真正的句子。", 0.05),
            (2.0, 4.0, prompt, 0.95),  # exact echo on silence
            (4.0, 6.0, "请使用标点符号", 0.85),  # substring echo on silence
            (6.0, 8.0, "另一句正常内容。", 0.10),
        ],
    )
    cfg = TranscriptionConfig(language="zh")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    segs, _ = transcribe(audio, cfg)
    assert [s.text for s in segs] == ["真正的句子。", "另一句正常内容。"]


def test_transcribe_keeps_prompt_substring_on_real_speech(monkeypatch, tmp_path: Path):
    """Segments matching prompt text but with low no_speech_prob are kept.

    Real speaker uttering '请使用标点符号' (a 7-char substring of the prompt)
    in a typing tutorial must survive — only silence-window hallucinations
    should be filtered.
    """
    _fake_whisper_pipeline(
        monkeypatch,
        segs=[
            (0.0, 2.0, "今天我们来讲打字。", 0.05),
            (2.0, 4.0, "请使用标点符号", 0.03),  # legitimate speech
            (4.0, 6.0, "普通话的句子", 0.10),  # legitimate speech
        ],
    )
    cfg = TranscriptionConfig(language="zh")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    segs, _ = transcribe(audio, cfg)
    assert [s.text for s in segs] == ["今天我们来讲打字。", "请使用标点符号", "普通话的句子"]


def test_transcribe_keeps_non_zh_segments_verbatim(monkeypatch, tmp_path: Path):
    """Non-zh path has no prompt, so echo-strip is a no-op even on high no_speech_prob."""
    _fake_whisper_pipeline(
        monkeypatch,
        segs=[(0.0, 2.0, "hello world", 0.9), (2.0, 4.0, "second", 0.05)],
    )
    cfg = TranscriptionConfig(language="en")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    segs, _ = transcribe(audio, cfg)
    assert [s.text for s in segs] == ["hello world", "second"]


def test_best_speaker_assigns_by_overlap():
    from bibilab.pipeline.diarize import SpeakerSegment
    from bibilab.pipeline.transcribe import WhisperSegment, _best_speaker

    seg = WhisperSegment(start=0.0, end=5.0, text="hello")
    speakers = [
        SpeakerSegment(start=0.0, end=3.0, speaker="SPK_0"),
        SpeakerSegment(start=3.0, end=10.0, speaker="SPK_1"),
    ]
    assert _best_speaker(seg, speakers) == "SPK_0"

    seg2 = WhisperSegment(start=6.0, end=9.0, text="world")
    assert _best_speaker(seg2, speakers) == "SPK_1"

    seg3 = WhisperSegment(start=20.0, end=25.0, text="alone")
    assert _best_speaker(seg3, speakers) is None


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
    # Telemetry: sentence_flushes must be credited (not miscredited to token=)
    assert "sentence=1" in caplog.text
    assert "token=0" in caplog.text


def test_chunk_no_sentence_end_flushes_at_target(caplog):
    """Without any sentence-end in buffer, token-flush bounds chunk at target."""
    filler = "word " * 25
    segs = [_seg(filler, start=float(i), end=float(i + 1)) for i in range(14)]

    with caplog.at_level("INFO", logger="bibilab.pipeline.chunk"):
        chunks = chunk_segments(segs, target_tokens=300)

    assert len(chunks) == 2
    # No sentence boundary anywhere → first chunk credited to token, not sentence.
    assert "token=1" in caplog.text
    assert "sentence=0" in caplog.text
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
    assert "sentence=1" in caplog.text


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
# _shared.py
# ---------------------------------------------------------------------------


def test_resolved_lang_with_ui_returns_ui_lang():
    assert _resolved_lang("ui", "zh") == "zh"
    assert _resolved_lang("ui", "en") == "en"


def test_resolved_lang_with_ui_falls_back_to_en():
    assert _resolved_lang("ui", None) == "en"


def test_resolved_lang_with_explicit_language():
    assert _resolved_lang("zh", None) == "zh"
    assert _resolved_lang("en", "zh") == "en"


# ---------------------------------------------------------------------------
# extract.py
# ---------------------------------------------------------------------------


def test_generate_overview_returns_overview_text(tmp_path: Path):
    from bibilab.config import AIConfig

    ai_cfg = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="en",
    )
    list_videos = [
        {"title": "Video A", "summary": "Introduction to ML concepts"},
        {"title": "Video B", "summary": "Deep learning architectures"},
    ]
    mock_overview = "This series introduces ML from basics to deep learning."
    with patch("bibilab.pipeline.extract._call_llm", return_value=mock_overview) as mock_call:
        result = generate_overview(list_videos, ai_cfg)

    assert result == mock_overview
    # Verify prompt includes both videos
    call_args = mock_call.call_args
    prompt = call_args[0][0]
    assert "Video A" in prompt
    assert "Video B" in prompt
    assert "Introduction to ML concepts" in prompt


def test_generate_overview_respects_output_language(tmp_path: Path):
    from bibilab.config import AIConfig

    ai_cfg = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="zh",
    )
    mock_overview = "本系列介绍机器学习基础"
    with patch("bibilab.pipeline.extract._call_llm", return_value=mock_overview) as mock_call:
        result = generate_overview([{"title": "Test", "summary": "Summary"}], ai_cfg, output_language="zh")

    assert result == mock_overview
    call_args = mock_call.call_args
    prompt = call_args[0][0]
    assert "请用中文回答" in prompt


# ---------------------------------------------------------------------------
# worker._generate_artifact language instruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_artifact_includes_zh_lang_instruction(tmp_path: Path, monkeypatch):
    """_generate_artifact prepends Chinese language instruction when ui_lang=zh."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.worker import WorkerLoop

    worker = WorkerLoop()

    captured_prompt = None

    def mock_call_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"name": "Test", "content": "# Test"}'

    monkeypatch.setattr("bibilab.worker._call_llm", mock_call_llm)

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="ui",
    )
    cfg.transcription.llm_timeout = 120
    cfg.transcription.llm_max_tokens = 2048
    cfg.ai.transcript_char_limit = 100000

    result = await worker._generate_artifact(
        prompt="Generate a summary",
        artifact_type="summary",
        transcript_text="This is a test transcript.",
        cfg=cfg,
        ui_lang="zh",
    )

    assert captured_prompt is not None
    assert captured_prompt.startswith("请用中文回答")
    assert "All output fields MUST be written in Chinese" in captured_prompt
    assert result.name == "Test"


@pytest.mark.asyncio
async def test_generate_artifact_includes_en_lang_instruction(tmp_path: Path, monkeypatch):
    """_generate_artifact prepends English language instruction when output_language=en."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.worker import WorkerLoop

    worker = WorkerLoop()

    captured_prompt = None

    def mock_call_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"name": "Test", "content": "# Test"}'

    monkeypatch.setattr("bibilab.worker._call_llm", mock_call_llm)

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="en",
    )
    cfg.transcription.llm_timeout = 120
    cfg.transcription.llm_max_tokens = 2048
    cfg.ai.transcript_char_limit = 100000

    await worker._generate_artifact(
        prompt="Generate a summary",
        artifact_type="summary",
        transcript_text="This is a test transcript.",
        cfg=cfg,
        ui_lang=None,
    )

    assert captured_prompt is not None
    assert captured_prompt.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured_prompt


@pytest.mark.asyncio
async def test_generate_artifact_unknown_lang_falls_back_to_english(tmp_path: Path, monkeypatch):
    """_generate_artifact with unrecognized output_language falls back to English."""
    from unittest.mock import MagicMock

    from bibilab.config import AIConfig
    from bibilab.worker import WorkerLoop

    worker = WorkerLoop()

    captured_prompt = None

    def mock_call_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"name": "Test", "content": "# Test"}'

    monkeypatch.setattr("bibilab.worker._call_llm", mock_call_llm)

    cfg = MagicMock()
    cfg.ai = AIConfig(
        protocol="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        output_language="fr",
    )
    cfg.transcription.llm_timeout = 120
    cfg.transcription.llm_max_tokens = 2048
    cfg.ai.transcript_char_limit = 100000

    await worker._generate_artifact(
        prompt="Generate a summary",
        artifact_type="summary",
        transcript_text="French biased transcript",
        cfg=cfg,
        ui_lang=None,
    )

    assert captured_prompt is not None
    assert captured_prompt.startswith("Respond in English only")
    assert "All output fields MUST be written in English" in captured_prompt
