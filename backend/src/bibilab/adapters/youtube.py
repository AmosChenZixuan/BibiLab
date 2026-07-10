"""YouTube platform adapter using yt-dlp. No credentials — public content only;
auth-walled videos surface as AuthRequiredError (cookie import is a separate issue)."""

import re
from pathlib import Path

import yt_dlp

from bibilab.adapters._ytdlp_common import (
    HTTP_RETRIES,
    SOCKET_TIMEOUT,
    apply_aria2c,
    gather_metadata,
    pick_thumbnail,
    safe_duration,
    strip_ansi,
)
from bibilab.adapters.base import (
    AuthRequiredError,
    DownloadError,
    PlatformAdapter,
    PlaylistMeta,
    VideoMeta,
)
from bibilab.config import downloads_dir

# Messages that mean "an account could see this": bot-check, private, age gate,
# members-only. Matched loosely against yt-dlp's DownloadError text.
_AUTH_RE = re.compile(r"sign\s*in|log\s*in|private|members[- ]only|confirm your age", re.IGNORECASE)

_WATCH_URL = "https://www.youtube.com/watch?v={}"


def _raise_mapped(exc: yt_dlp.utils.DownloadError) -> None:
    msg = str(exc)
    if _AUTH_RE.search(msg):
        raise AuthRequiredError("video") from exc
    raise DownloadError(strip_ansi(msg)) from exc


def _entry_to_video_meta(entry: dict) -> VideoMeta | None:
    """None when the entry carries no id — an id-less VideoMeta would corrupt dedup."""
    vid = entry.get("id")
    if not vid:
        return None
    return VideoMeta(
        video_id=vid,
        title=entry.get("title") or "Untitled",
        platform="youtube",
        source_url=entry.get("webpage_url") or entry.get("url") or _WATCH_URL.format(vid),
        cover_url=pick_thumbnail(entry),
        duration_seconds=safe_duration(entry.get("duration")),
        uploader=entry.get("uploader") or entry.get("channel") or "",
    )


class YouTubeAdapter(PlatformAdapter):
    def resolve_flat(self, url: str) -> PlaylistMeta:
        opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            _raise_mapped(exc)

        if info.get("_type") == "playlist":
            entries = info.get("entries") or []
            title = info.get("title", "Untitled Playlist")
        else:
            entries = [info]
            title = info.get("title", "Untitled")
        videos = [vm for e in entries if (vm := _entry_to_video_meta(e)) is not None]

        return PlaylistMeta(
            playlist_id=info.get("id", url),
            title=title,
            platform="youtube",
            source_url=url,
            videos=videos,
        )

    async def get_videos_metadata(self, video_ids: list[str]) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]:
        def fetch_one(vid: str) -> VideoMeta | None:
            opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(_WATCH_URL.format(vid), download=False)
            except Exception:
                # One bad video must not sink the batch — the id is simply
                # omitted from the result map (same contract as bilibili).
                return None
            return _entry_to_video_meta(info)

        return (await gather_metadata(video_ids, fetch_one), {})

    def download(self, video_id: str, source_url: str, connections: int) -> Path:
        out_dir = downloads_dir()
        opts: dict = {
            "quiet": False,
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
            "format": "bestaudio/best",
            "retries": HTTP_RETRIES,
            "socket_timeout": SOCKET_TIMEOUT,
        }
        apply_aria2c(opts, connections)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            _raise_mapped(exc)

        return out_dir / f"{video_id}.{info.get('ext', 'mp4')}"
