"""FFmpeg audio extraction step."""

import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

# Minimum fraction of the reference duration the decoded audio must cover.
# Healthy extractions land ~0.95+ (trailing end-silence + rounding); below this
# the audio is incomplete or corrupt.
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


def _has_audio_stream(path: Path) -> bool:
    """True if the file has an audio stream; True on probe failure too —
    an unreadable file should fail in extraction with FFmpeg's own error,
    not be misreported as audio-less."""
    try:
        streams = ffmpeg.probe(str(path))["streams"]
    except (ffmpeg.Error, KeyError, TypeError):
        return True
    return any(s.get("codec_type") == "audio" for s in streams)


def _validate_audio_duration(decoded: float, container: float, expected: float) -> None:
    """Fail loud when decoded audio is too short against a known reference.

    A byte-truncated faststart m4a still decodes to a short wav with ffmpeg
    exiting 0 (the leading ``moov`` keeps the container parseable), so duration
    must be checked explicitly rather than trusting the exit code. Two
    independent references, either of which fires:

    - ``container``: the input file's own reported duration — catches ffmpeg
      dropping audio even when the authoritative length is unknown.
    - ``expected``: the platform-reported duration — catches a file that is
      itself short (its container agrees with the truncated decode).

    A reference of ``0`` means "unknown" and is skipped. When *both* are unknown
    nothing can be validated, so the skip is logged rather than passing silently
    (a truncated file would otherwise slip through undetected).
    """
    checked = False
    for label, ref in (("container", container), ("expected", expected)):
        if ref <= 0:
            continue
        checked = True
        if decoded < ref * _DURATION_FLOOR:
            raise PipelineError(
                f"audio_truncated: decoded audio {decoded:.1f}s is below "
                f"{_DURATION_FLOOR:.0%} of {label} duration {ref:.1f}s — "
                f"incomplete or corrupt download"
            )
    if not checked:
        logger.warning(
            "Audio duration unverified: no container or expected reference "
            "available (decoded %.1fs) — a truncated file cannot be detected here",
            decoded,
        )


def extract_audio(video_path: Path, expected_duration: float = 0.0) -> Path:
    """Extract 16kHz mono WAV from video_path; delete source video on success.

    Validates the decoded duration against the input container and the
    platform-reported ``expected_duration`` (seconds, ``0`` = unknown) and
    raises ``PipelineError`` on a short/corrupt result instead of persisting a
    silently truncated transcript. Returns the path to the extracted WAV file.
    """
    if not _has_audio_stream(video_path):
        # e.g. a platform variant shipped without an audio track: fail with
        # a clear message instead of FFmpeg's raw "does not contain any
        # stream" dump from the extraction below.
        raise PipelineError("video has no audio track")

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
