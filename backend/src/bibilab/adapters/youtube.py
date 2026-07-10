"""YouTube platform adapter using yt-dlp. No credentials — public content only;
auth-walled videos surface as AuthRequiredError (cookie import is a separate issue)."""

import asyncio
import re
import shutil
from pathlib import Path

import yt_dlp

from bibilab.adapters.base import (
    AuthRequiredError,
    DownloadError,
    PlatformAdapter,
    PlaylistMeta,
    VideoMeta,
)
from bibilab.config import bibilab_home

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Messages that mean "an account could see this": bot-check, private, age gate,
# members-only. Matched loosely against yt-dlp's DownloadError text.
_AUTH_RE = re.compile(r"sign\s*in|log\s*in|private|members[- ]only|confirm your age", re.IGNORECASE)

_METADATA_CONCURRENCY = 8
_HTTP_RETRIES = 10
_SOCKET_TIMEOUT = 60

_WATCH_URL = "https://www.youtube.com/watch?v={}"


def _raise_mapped(exc: yt_dlp.utils.DownloadError) -> None:
    msg = str(exc)
    if _AUTH_RE.search(msg):
        raise AuthRequiredError("video") from exc
    raise DownloadError(_ANSI_RE.sub("", msg)) from exc


def _flat_thumbnail(entry: dict) -> str:
    if entry.get("thumbnail"):
        return entry["thumbnail"]
    thumbnails = entry.get("thumbnails") or []
    return (thumbnails[-1].get("url", "") or "") if thumbnails else ""


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
        cover_url=_flat_thumbnail(entry),
        duration_seconds=int(entry.get("duration") or 0),
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
        if not video_ids:
            return ({}, {})

        semaphore = asyncio.Semaphore(_METADATA_CONCURRENCY)

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

        async def fetch_bounded(vid: str) -> VideoMeta | None:
            async with semaphore:
                return await asyncio.to_thread(fetch_one, vid)

        results = await asyncio.gather(*[fetch_bounded(vid) for vid in video_ids])
        return ({vid: meta for vid, meta in zip(video_ids, results) if meta is not None}, {})

    def download(self, video_id: str, source_url: str, connections: int) -> Path:
        downloads_dir = bibilab_home() / "downloads"
        opts: dict = {
            "quiet": False,
            "outtmpl": str(downloads_dir / f"{video_id}.%(ext)s"),
            "format": "bestaudio/best",
            "retries": _HTTP_RETRIES,
            "socket_timeout": _SOCKET_TIMEOUT,
        }
        if shutil.which("aria2c"):
            opts["external_downloader"] = "aria2c"
            opts["external_downloader_args"] = {
                "aria2c": [f"-x{connections}", f"-s{connections}", "-k1M", "--file-allocation=none"],
            }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            _raise_mapped(exc)

        return downloads_dir / f"{video_id}.{info.get('ext', 'mp4')}"
