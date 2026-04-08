"""Unit tests for pipeline modules (mocked I/O and LLM)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.pipeline._shared import _resolved_lang
from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import RagChunk, chunk_segments
from bibilab.pipeline.extract import generate_overview
from bibilab.pipeline.transcribe import (
    WhisperSegment,
    _compute_type_for_device,
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
        provider="openai",
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
        provider="openai",
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
