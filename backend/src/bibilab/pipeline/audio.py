"""FFmpeg audio extraction step."""

import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


def extract_audio(video_path: Path) -> Path:
    """Extract 16kHz mono WAV from video_path; delete source video on success.

    Returns the path to the extracted WAV file.
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

    video_path.unlink(missing_ok=True)
    logger.info("Extracted audio to %s", wav_path)
    return wav_path
