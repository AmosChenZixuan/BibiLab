"""FFmpeg audio extraction step."""

import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

# Minimum fraction of the reference duration the decoded audio must cover.
# Healthy extractions land ~0.96 (trailing end-silence + rounding); below this
# the audio is incomplete or corrupt. See issue #546.
_DURATION_FLOOR = 0.9


class PipelineError(Exception):
    pass


def _probe_duration(path: Path) -> float:
    """Container/stream duration in seconds, or 0.0 if it can't be determined."""
    try:
        info = ffmpeg.probe(str(path))
        return float(info["format"]["duration"])
    except (ffmpeg.Error, KeyError, ValueError, TypeError):
        return 0.0


def _validate_audio_duration(decoded: float, container: float, expected: float) -> None:
    """Fail loud when decoded audio is too short against a known reference.

    A damaged faststart m4a makes ffmpeg exit 0 with a short wav (the front
    ``moov`` keeps the container parseable), so length must be checked
    explicitly. Two independent references, either of which fires:

    - ``container``: the input m4a's own duration — catches ffmpeg silently
      dropping audio even when the authoritative length is unknown.
    - ``expected``: the platform-reported ``duration_seconds`` — catches a file
      that is itself short (its container agrees with the truncated decode).

    Unknown references (``0``) are skipped (see issue #546 zero-duration guard).
    """
    for label, ref in (("container", container), ("expected", expected)):
        if ref > 0 and decoded < ref * _DURATION_FLOOR:
            raise PipelineError(
                f"audio_truncated: decoded audio {decoded:.1f}s is below "
                f"{_DURATION_FLOOR:.0%} of {label} duration {ref:.1f}s — "
                f"incomplete or corrupt download"
            )


def extract_audio(video_path: Path, expected_duration: float = 0.0) -> Path:
    """Extract 16kHz mono WAV from video_path; delete source video on success.

    Validates the decoded duration against the input container and the
    platform-reported ``expected_duration`` (seconds, ``0`` = unknown) and
    raises ``PipelineError`` on a short/corrupt result instead of persisting a
    silently truncated transcript. Returns the path to the extracted WAV file.
    """
    wav_path = video_path.with_suffix(".wav")
    try:
        (
            ffmpeg.input(str(video_path))
            .output(
                str(wav_path),
                ar=16000,
                ac=1,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        msg = f"FFmpeg audio extraction failed: {exc.stderr.decode()!r}"
        raise PipelineError(msg) from exc

    _validate_audio_duration(
        decoded=_probe_duration(wav_path),
        container=_probe_duration(video_path),
        expected=expected_duration,
    )

    video_path.unlink(missing_ok=True)
    logger.info("Extracted audio to %s", wav_path)
    return wav_path
