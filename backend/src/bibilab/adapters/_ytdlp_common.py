"""Shared yt-dlp plumbing for platform adapters (three consumers: bilibili, youtube, tiktok)."""

import asyncio
import re
import shutil
from collections.abc import Callable
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


def strip_ansi(message: str) -> str:
    """yt-dlp error strings may embed terminal color codes."""
    return _ANSI_RE.sub("", message)


def apply_aria2c(opts: dict, connections: int) -> None:
    """Route the download through aria2c when available (parallel connections
    sidestep per-IP throttles); absent-aria2c leaves opts untouched — the
    native downloader still works, just slower under throttle."""
    if shutil.which("aria2c"):
        opts["external_downloader"] = "aria2c"
        opts["external_downloader_args"] = {
            "aria2c": [f"-x{connections}", f"-s{connections}", "-k1M", "--file-allocation=none"],
        }


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
    exc: Exception,
    auth_re: re.Pattern,
    *,
    message_overrides: tuple[tuple[re.Pattern, str], ...] = (),
    hint: str = "",
) -> NoReturn:
    """Map a yt-dlp DownloadError to the domain errors: auth-family messages →
    AuthRequiredError; an override pattern → DownloadError with its fixed
    message; anything else → DownloadError (ANSI stripped, optional hint).
    bilibili keeps its own inline mapping — its lowercased matching, 412
    handling and cookie revalidation don't fit this shape."""
    msg = str(exc)
    if auth_re.search(msg):
        raise AuthRequiredError("video") from exc
    for pattern, override in message_overrides:
        if pattern.search(msg):
            raise DownloadError(override) from exc
    raise DownloadError(strip_ansi(msg) + hint) from exc


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
