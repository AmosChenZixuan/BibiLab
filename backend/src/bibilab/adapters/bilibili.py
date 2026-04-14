"""Bilibili platform adapter using yt-dlp."""

import asyncio
import logging
import re
from pathlib import Path

import httpx
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
_PLAYLIST_RE = re.compile(
    r"bilibili\.com/medialist|space\.bilibili\.com/\d+/channel|space\.bilibili\.com/\d+/lists", re.IGNORECASE
)
_COURSE_RE = re.compile(r"bilibili\.com/cheese", re.IGNORECASE)

# Strip ANSI escape codes from yt_dlp output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_METADATA_CONCURRENCY = 8


def _resource_type(url: str) -> str:
    if _COURSE_RE.search(url):
        return "course"
    if _PLAYLIST_RE.search(url):
        return "playlist"
    return "video"


def _ydl_opts(cookie: str, quiet: bool = True, extract_flat: bool | str = False) -> dict:
    opts: dict = {
        "quiet": quiet,
        "no_warnings": quiet,
        "extract_flat": extract_flat,
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
            if info.get("_type") == "playlist":
                return self._resolve_playlist(info)
            return _info_to_video_meta(info)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
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

    def resolve_flat(self, url: str) -> PlaylistMeta:
        rtype = _resource_type(url)

        if rtype == "video":
            opts = _ydl_opts(self._cookie)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as exc:
                raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc

            playlist_id = info.get("id", url)
            title = info.get("title", "Untitled")
            vm = _info_to_video_meta(info)
            return PlaylistMeta(
                playlist_id=playlist_id,
                title=title,
                platform="bilibili",
                source_url=url,
                videos=[vm],
            )

        if rtype == "course":
            raise AuthRequiredError("course")

        opts = _ydl_opts(self._cookie, extract_flat="in_playlist")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc

        playlist_id = info.get("id", url)
        title = info.get("title", "Untitled Playlist")
        entries = info.get("entries") or []

        videos = []
        for e in entries:
            if not e.get("id"):
                continue
            base_id, part_num = _split_video_id(e.get("id", ""))
            video_id = f"{base_id}_p{part_num}" if part_num is not None else base_id
            vm = VideoMeta(
                video_id=video_id,
                title=e.get("title", "Untitled") or "Untitled",
                platform="bilibili",
                source_url=e.get("url", "") or e.get("webpage_url", ""),
                cover_url=e.get("thumbnail", "") or "",
                duration_seconds=0,
                uploader="",
                part_label=f"P{part_num}" if part_num is not None else None,
            )
            videos.append(vm)

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
            base_id, part_num = _split_video_id(e.get("id", ""))
            if part_num is not None:
                e["id"] = f"{base_id}_p{part_num}"
            vm = _info_to_video_meta(e, fallback_uploader=base_uploader)
            if part_num is not None:
                vm.part_label = f"P{part_num}"
            videos.append(vm)

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
        if part_num is not None:
            opts["playlist_items"] = str(part_num)

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

    async def get_videos_metadata(self, video_ids: list[str]) -> dict[str, VideoMeta]:
        if not video_ids:
            return {}

        unique_bvids: dict[str, list[str]] = {}
        for vid in video_ids:
            bvid, _ = _split_video_id(vid)
            unique_bvids.setdefault(bvid, []).append(vid)

        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.bilibili.com",
        }
        if self._cookie:
            headers["Cookie"] = self._cookie

        semaphore = asyncio.Semaphore(_METADATA_CONCURRENCY)

        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:

            async def fetch_one(bvid: str) -> tuple[str, VideoMeta | None]:
                async with semaphore:
                    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            return (bvid, None)
                        json_data = resp.json()
                    except httpx.HTTPError:
                        return (bvid, None)

                    data = json_data.get("data")
                    if not isinstance(data, dict):
                        return (bvid, None)

                    return (
                        bvid,
                        VideoMeta(
                            video_id=bvid,
                            title=data.get("title", "Untitled") or "Untitled",
                            platform="bilibili",
                            source_url=data.get("short_link_v1", f"https://bilibili.com/video/{bvid}"),
                            cover_url=data.get("pic", "") or "",
                            duration_seconds=int(data.get("duration", 0) or 0),
                            uploader=data.get("owner", {}).get("name", "") or "",
                        ),
                    )

            results = await asyncio.gather(*[fetch_one(bvid) for bvid in unique_bvids])

        result: dict[str, VideoMeta] = {}
        for bvid, meta in results:
            if meta is not None:
                for orig_id in unique_bvids[bvid]:
                    result[orig_id] = meta
        return result
