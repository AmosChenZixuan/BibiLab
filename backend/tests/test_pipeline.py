"""Unit tests for pipeline modules (mocked I/O and LLM)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.pipeline.audio import PipelineError, extract_audio
from bibilab.pipeline.chunk import RagChunk, chunk_segments
from bibilab.pipeline.extract import (
    _parse_response,
    extract_knowledge,
)
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
# extract.py
# ---------------------------------------------------------------------------


def test_parse_response_clean_json():
    raw = '{"title": "T", "summary": "S", "key_points": [{"timestamp": "[00:01:00]", "text": "A"}]}'
    result = _parse_response(raw)
    assert result.title == "T"
    assert result.summary == "S"
    assert result.key_points[0].timestamp == "[00:01:00]"


def test_parse_response_strips_code_fence():
    raw = '```json\n{"title": "T", "summary": "S", "key_points": []}\n```'
    result = _parse_response(raw)
    assert result.title == "T"


def test_extract_knowledge_calls_llm(tmp_path: Path):
    from bibilab.adapters.base import VideoMeta
    from bibilab.config import AIConfig

    meta = VideoMeta(
        video_id="BV1",
        title="Test Video",
        platform="bilibili",
        source_url="https://example.com",
        cover_url="",
        duration_seconds=100,
        uploader="user",
    )
    cfg = AIConfig(provider="openai", model="gpt-4o", api_key="fake")
    good_response = '{"title": "LLM Title", "summary": "Sum", "key_points": []}'

    with patch("bibilab.pipeline.extract._call_llm", return_value=good_response):
        result = extract_knowledge("some transcript", meta, cfg)

    assert result.title == "LLM Title"
    assert result.summary == "Sum"


def test_extract_knowledge_retries_on_bad_json():
    from bibilab.adapters.base import VideoMeta
    from bibilab.config import AIConfig

    meta = VideoMeta("BV1", "T", "bilibili", "https://x.com", "", 0, "")
    cfg = AIConfig(provider="openai", model="gpt-4o", api_key="fake")
    good = '{"title": "T2", "summary": "S2", "key_points": []}'

    with patch("bibilab.pipeline.extract._call_llm", side_effect=["not json at all", good]):
        result = extract_knowledge("transcript", meta, cfg)

    assert result.title == "T2"
