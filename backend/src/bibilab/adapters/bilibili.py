"""Bilibili platform adapter using yt-dlp."""

import logging
import re
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

logger = logging.getLogger(__name__)

# URL patterns for Bilibili resource types
_PLAYLIST_RE = re.compile(r"bilibili\.com/medialist|space\.bilibili\.com/\d+/channel", re.IGNORECASE)
_COURSE_RE = re.compile(r"bilibili\.com/cheese", re.IGNORECASE)

# Strip ANSI escape codes from yt_dlp output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _resource_type(url: str) -> str:
    if _COURSE_RE.search(url):
        return "course"
    if _PLAYLIST_RE.search(url):
        return "playlist"
    return "video"


def _ydl_opts(cookie: str, quiet: bool = True) -> dict:
    opts: dict = {
        "quiet": quiet,
        "no_warnings": quiet,
        "extract_flat": False,
    }
    if cookie:
        opts["http_headers"] = {"Cookie": cookie}
    return opts


def _split_video_id(video_id: str) -> tuple[str, int | None]:
    """Split a video_id like 'BVxxx_p3' into ('BVxxx', 3). Returns (video_id, None) if no part."""
    m = re.match(r"^(.+?)_p(\d+)$", video_id)
    if m:
        return m.group(1), int(m.group(2))
    return video_id, None


def _info_to_video_meta(info: dict, platform: str = "bilibili", fallback_uploader: str = "") -> VideoMeta:
    video_id = info.get("id", "")
    playlist_index = info.get("playlist_index")
    part_label = f"P{playlist_index}" if playlist_index else None
    return VideoMeta(
        video_id=video_id,
        title=info.get("title", "Untitled"),
        platform=platform,
        source_url=info.get("webpage_url", info.get("url", "")),
        cover_url=info.get("thumbnail", ""),
        duration_seconds=int(info.get("duration", 0) or 0),
        uploader=info.get("uploader", fallback_uploader),
        part_label=part_label,
    )


class BilibiliAdapter(PlatformAdapter):
    def __init__(self, cookie: str = "") -> None:
        self._cookie = cookie

    def requires_auth(self, resource_type: str) -> bool:
        return resource_type == "course"

    def resolve(self, url: str) -> VideoMeta | PlaylistMeta:
        rtype = _resource_type(url)

        opts = _ydl_opts(self._cookie)

        if rtype == "video":
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as exc:
                raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc
            if info.get("_type") == "playlist":
                return self._resolve_playlist(info)
            return _info_to_video_meta(info)

        # Playlist or course: use flat extraction to get the list of entries
        # without downloading metadata for each video individually.
        # Full per-video metadata is fetched lazily during the pipeline.
        flat_opts = {**opts, "extract_flat": True}
        try:
            with yt_dlp.YoutubeDL(flat_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc

        playlist_id = info.get("id", url)
        title = info.get("title", "Untitled Playlist")
        entries = info.get("entries") or []

        videos = [_info_to_video_meta(e, fallback_uploader=info.get("uploader", "")) for e in entries if e.get("id")]

        return PlaylistMeta(
            playlist_id=playlist_id,
            title=title,
            platform="bilibili",
            source_url=url,
            videos=videos,
        )

    def _resolve_playlist(self, info: dict) -> PlaylistMeta:
        playlist_id = info.get("id", "")
        title = info.get("title", "Untitled Playlist")
        entries = info.get("entries") or []
        base_uploader = info.get("uploader", "")

        videos = []
        for e in entries:
            if not e.get("id"):
                continue
            entry_id = e.get("id", "")
            base_id, part_num = _split_video_id(entry_id)
            if part_num is not None:
                e["id"] = f"{base_id}_p{part_num}"
            videos.append(_info_to_video_meta(e, fallback_uploader=base_uploader))

        return PlaylistMeta(
            playlist_id=playlist_id,
            title=title,
            platform="bilibili",
            source_url=info.get("webpage_url", ""),
            videos=videos,
        )

    def download(self, video_id: str, source_url: str) -> Path:
        downloads_dir = bibilab_home() / "downloads"
        output_template = str(downloads_dir / f"{video_id}.%(ext)s")
        opts = {
            **_ydl_opts(self._cookie, quiet=False),
            "outtmpl": output_template,
            "format": "bestvideo+bestaudio/best",
        }

        _, part_num = _split_video_id(video_id)
        extra_info = {}
        if part_num is not None:
            extra_info["playlist_items"] = str(part_num)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True, extra_info=extra_info or None)
                ext = info.get("ext", "mp4")
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc).lower()
            if "login" in msg or "sign in" in msg or "403" in msg:
                raise AuthRequiredError("video") from exc
            raise

        return downloads_dir / f"{video_id}.{ext}"
