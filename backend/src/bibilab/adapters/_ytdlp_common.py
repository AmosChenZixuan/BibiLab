"""Shared yt-dlp plumbing for platform adapters (three consumers: bilibili, youtube, tiktok)."""

import asyncio
import contextlib
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, TypeVar

from bibilab.adapters.base import AuthRequiredError, DownloadError

_T = TypeVar("_T")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Per-video metadata fetch parallelism; polite to every platform's web API.
METADATA_CONCURRENCY = 8
# Per-request retries: resumes from the .part via Range, cheap to keep high.
HTTP_RETRIES = 10
# Per-read socket timeout (s): trips only on a true stall, turning a silent
# hang into a retriable error instead of wedging the serialized stage.
SOCKET_TIMEOUT = 60
# Grace period after SIGTERM before escalating to SIGKILL, so a child that
# traps or ignores SIGTERM cannot outlive a cancel indefinitely.
TERMINATE_TIMEOUT = 5.0


def strip_ansi(message: str) -> str:
    """yt-dlp error strings may embed terminal color codes."""
    return _ANSI_RE.sub("", message)


def aria2c_argv(connections: int) -> list[str]:
    """CLI flags that route the download through aria2c when available (parallel
    connections sidestep per-IP throttles); absent-aria2c returns no flags — the
    native downloader still works, just slower under throttle."""
    if not shutil.which("aria2c"):
        return []
    return [
        "--downloader",
        "aria2c",
        "--downloader-args",
        f"aria2c:-x{connections} -s{connections} -k1M --file-allocation=none",
    ]


async def _run_subprocess(argv: list[str], *, terminate_timeout: float = TERMINATE_TIMEOUT) -> tuple[str, str, int]:
    """Run argv as a child process and return (stdout, stderr, returncode).

    A cancel while awaiting the child's exit terminates it (SIGTERM) and
    re-raises CancelledError without reading the child's exit status — a
    non-zero exit caused by our own termination must never be mistaken for a
    real failure. A child that doesn't exit within terminate_timeout is
    escalated to SIGKILL so a wedged or SIGTERM-ignoring process can't outlive
    the cancel.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=terminate_timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        raise
    return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode


async def run_ytdlp(args: list[str], *, terminate_timeout: float = TERMINATE_TIMEOUT) -> tuple[str, str, int]:
    """Run yt-dlp as a child of this interpreter (never a bare `yt-dlp` binary,
    which may not be on PATH in a container or a uv-managed venv) and return
    (stdout, stderr, returncode)."""
    return await _run_subprocess([sys.executable, "-m", "yt_dlp", *args], terminate_timeout=terminate_timeout)


def parse_download_path(stdout: str) -> Path:
    """Extract the output path from `--print after_move:filepath` output.

    yt-dlp writes exactly one such line per completed download. Never glob
    the downloads directory instead: `.part` residue from a prior attempt
    shares the same filename pattern as the real output, so a glob can
    silently return the wrong file.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line:
            return Path(line)
    raise DownloadError("yt-dlp completed without reporting an output path")


async def gather_metadata(video_ids: list[str], fetch_one: Callable[[str], _T | None]) -> dict[str, _T]:
    """Run a blocking per-id fetch across a thread pool with bounded
    concurrency; failed ids (fetch_one returns None) are omitted."""
    semaphore = asyncio.Semaphore(METADATA_CONCURRENCY)

    async def fetch_bounded(vid: str) -> _T | None:
        async with semaphore:
            return await asyncio.to_thread(fetch_one, vid)

    results = await asyncio.gather(*[fetch_bounded(vid) for vid in video_ids])
    return {vid: meta for vid, meta in zip(video_ids, results) if meta is not None}


def raise_mapped(
    message: str,
    auth_re: re.Pattern,
    *,
    message_overrides: tuple[tuple[re.Pattern, str], ...] = (),
    hint: str = "",
    cause: Exception | None = None,
) -> NoReturn:
    """Map a yt-dlp error message (an exception's str(), or a subprocess's
    stderr) to the domain errors: auth-family messages → AuthRequiredError;
    an override pattern → DownloadError with its fixed message; anything else
    → DownloadError (ANSI stripped, optional hint). bilibili keeps its own
    inline mapping — its lowercased matching, 412 handling and cookie
    revalidation don't fit this shape."""
    if auth_re.search(message):
        raise AuthRequiredError("video") from cause
    for pattern, override in message_overrides:
        if pattern.search(message):
            raise DownloadError(override) from cause
    raise DownloadError(strip_ansi(message) + hint) from cause


def safe_duration(value) -> int:
    """yt-dlp durations are usually numeric, but the field contract allows
    strings some extractors emit ('mm:ss' etc.); one bad value must not
    sink the whole entry list — degrade to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def pick_thumbnail(entry: dict) -> str:
    """Best thumbnail URL from a yt-dlp info dict. Prefers the singular
    `thumbnail`; falls back to the largest-area entry of `thumbnails` —
    list order is undocumented for flat-playlist entries, so never index it."""
    if entry.get("thumbnail"):
        return entry["thumbnail"]
    thumbs = [t for t in (entry.get("thumbnails") or []) if t.get("url")]
    if not thumbs:
        return ""
    best = max(thumbs, key=lambda t: (t.get("width") or 0) * (t.get("height") or 0))
    if not (best.get("width") or best.get("height")):
        # No entry carries dimensions — fall back to the last (yt-dlp sorts
        # preference ascending where it sorts at all).
        return thumbs[-1]["url"]
    return best["url"]
