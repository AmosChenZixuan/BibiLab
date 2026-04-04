"""Bilibili platform adapter using yt-dlp."""

import logging
import re
from pathlib import Path

import yt_dlp

from locus.adapters.base import (
    AuthRequiredError,
    DownloadError,
    PlatformAdapter,
    PlaylistMeta,
    VideoMeta,
)
from locus.config import locus_home

logger = logging.getLogger(__name__)

# URL patterns for Bilibili resource types
_PLAYLIST_RE = re.compile(
    r"bilibili\.com/medialist|space\.bilibili\.com/\d+/channel", re.IGNORECASE
)
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


def _info_to_video_meta(
    info: dict, platform: str = "bilibili", fallback_uploader: str = ""
) -> VideoMeta:
    return VideoMeta(
        video_id=info.get("id", ""),
        title=info.get("title", "Untitled"),
        platform=platform,
        source_url=info.get("webpage_url", info.get("url", "")),
        cover_url=info.get("thumbnail", ""),
        duration_seconds=int(info.get("duration", 0) or 0),
        uploader=info.get("uploader", fallback_uploader),
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

        videos = [
            _info_to_video_meta(e, fallback_uploader=info.get("uploader", ""))
            for e in entries
            if e.get("id")
        ]

        return PlaylistMeta(
            playlist_id=playlist_id,
            title=title,
            platform="bilibili",
            source_url=url,
            videos=videos,
        )

    def download(self, video_id: str, source_url: str) -> Path:
        downloads_dir = locus_home() / "downloads"
        output_template = str(downloads_dir / f"{video_id}.%(ext)s")
        opts = {
            **_ydl_opts(self._cookie, quiet=False),
            "outtmpl": output_template,
            "format": "bestvideo+bestaudio/best",
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
                ext = info.get("ext", "mp4")
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc).lower()
            if "login" in msg or "sign in" in msg or "403" in msg:
                raise AuthRequiredError("video") from exc
            raise

        return downloads_dir / f"{video_id}.{ext}"
