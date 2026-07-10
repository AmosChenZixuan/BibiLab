"""Shared yt-dlp plumbing for platform adapters (three consumers: bilibili, youtube, tiktok)."""

import re
import shutil

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
