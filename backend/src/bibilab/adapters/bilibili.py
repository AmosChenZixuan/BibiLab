"""Bilibili platform adapter using yt-dlp."""

import asyncio
import logging
import re
import shutil
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
from bibilab.config import bibilab_home, load_config

logger = logging.getLogger(__name__)

# URL patterns for Bilibili resource types
_PLAYLIST_RE = re.compile(
    r"bilibili\.com/medialist|space\.bilibili\.com/\d+/channel|space\.bilibili\.com/\d+/lists|space\.bilibili\.com/\d+/favlist",
    re.IGNORECASE,
)
_COURSE_RE = re.compile(r"bilibili\.com/cheese", re.IGNORECASE)

# Strip ANSI escape codes from yt_dlp output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_AUTH_RE = re.compile(r"log\s*in|sign\s*in|403", re.IGNORECASE)
# Multi-part page selector in a video '?p=N' url
_PART_RE = re.compile(r"[?&]p=(\d+)")

_METADATA_CONCURRENCY = 8
# Parallel DASH-fragment downloads per video (audio is served fragmented).
# Modest so N jobs × this stays friendly to Bilibili rate limits.
_FRAGMENT_CONCURRENCY = 4
# Per-fragment HTTP retries. yt-dlp's bare opts default to 0 (None → RetryManager
# short-circuits), so transient CDN timeouts propagate as fatal DownloadError.
_FRAGMENT_RETRIES = 5
# Per-request retries for the contiguous audio download. Each retry resumes from
# the .part via Range (no re-download, no sleep), so a high budget cheaply rides
# out bilibili's per-IP throttle dropping long connections mid-stream.
_HTTP_RETRIES = 10
# Per-read socket timeout (s). Slow-but-progressing transfers reset it on each
# read; it only trips on a true stall, turning a silent hang into a retriable
# error that resumes from the .part instead of wedging the serialized stage.
_SOCKET_TIMEOUT = 60
_BILIBILI_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_cookie_file_cache: tuple[str, Path] | None = None


def _resource_type(url: str) -> str:
    if _COURSE_RE.search(url):
        return "course"
    if _PLAYLIST_RE.search(url):
        return "playlist"
    return "video"


def _cookie_file(cookie: str) -> str | None:
    if not cookie:
        return None
    global _cookie_file_cache
    if _cookie_file_cache is not None and _cookie_file_cache[0] == cookie:
        return str(_cookie_file_cache[1])
    path = bibilab_home() / "bilibili_cookies.txt"
    lines = ["# Netscape HTTP Cookie File"]
    for pair in cookie.split("; "):
        key, _, val = pair.partition("=")
        if key:
            lines.append(f".bilibili.com\tTRUE\t/\tFALSE\t0\t{key}\t{val}")
    path.write_text("\n".join(lines) + "\n")
    _cookie_file_cache = (cookie, path)
    return str(path)


def _ydl_opts(cookie: str, quiet: bool = True, extract_flat: bool | str = False) -> dict:
    opts: dict = {
        "quiet": quiet,
        "no_warnings": quiet,
        "extract_flat": extract_flat,
    }
    cf = _cookie_file(cookie)
    if cf:
        opts["cookiefile"] = cf
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

    async def _validate_cookies(self) -> bool:
        """Check if configured cookies are still valid via the bilibili nav API."""
        if not self._cookie:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    _BILIBILI_NAV_URL,
                    headers={"Cookie": self._cookie, "Referer": "https://www.bilibili.com/"},
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def resolve_flat(self, url: str) -> PlaylistMeta:
        rtype = _resource_type(url)

        if rtype == "video":
            opts = _ydl_opts(self._cookie, extract_flat="in_playlist")
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as exc:
                msg = str(exc).lower()
                if _AUTH_RE.search(msg):
                    raise AuthRequiredError("video") from exc
                raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc

            if info.get("_type") == "playlist":
                return self._resolve_playlist(info)

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
            msg = str(exc).lower()
            if _AUTH_RE.search(msg):
                raise AuthRequiredError("playlist") from exc
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
            eid = e.get("id")
            if eid:
                base_id, part_num = _split_video_id(eid)
            else:
                # extract_flat entries carry no id; derive the part from the '?p=N' url
                base_id = playlist_id
                m = _PART_RE.search(e.get("url") or "")
                part_num = int(m.group(1)) if m else None
            if part_num is None:
                continue
            videos.append(
                VideoMeta(
                    video_id=f"{base_id}_p{part_num}",
                    title=e.get("title") or title,
                    platform="bilibili",
                    source_url=e.get("url")
                    or e.get("webpage_url")
                    or f"https://www.bilibili.com/video/{base_id}?p={part_num}",
                    cover_url=e.get("thumbnail") or "",
                    duration_seconds=int(e.get("duration") or 0),
                    uploader=e.get("uploader") or base_uploader,
                    part_label=f"P{part_num}",
                )
            )

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
            "format": "bestaudio/best",
            "concurrent_fragment_downloads": _FRAGMENT_CONCURRENCY,
            "retries": _HTTP_RETRIES,
            "fragment_retries": _FRAGMENT_RETRIES,
            "socket_timeout": _SOCKET_TIMEOUT,
        }
        # Route through aria2c when available. It bounds the per-connection
        # throttle tail (master-native hits ~0.85 MB/s on a single connection
        # 30% of downloads → 64 s for a 64 MB file; aria2c -x16 holds max 5.7 s
        # in the same window). On aria2c-absent hosts we fall back to native
        # yt-dlp above and ingest still works, just slower.
        if shutil.which("aria2c"):
            n = load_config().backend.download_connections
            opts["external_downloader"] = "aria2c"
            opts["external_downloader_args"] = {
                "aria2c": [f"-x{n}", f"-s{n}", "-k1M", "--file-allocation=none"],
            }

        _, part_num = _split_video_id(video_id)
        if part_num is not None:
            opts["playlist_items"] = str(part_num)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc).lower()
            if _AUTH_RE.search(msg) or "412" in msg:
                raise AuthRequiredError("video") from exc
            raise DownloadError(_ANSI_RE.sub("", str(exc))) from exc

        return downloads_dir / f"{video_id}.{info.get('ext', 'mp4')}"

    async def get_videos_metadata(self, video_ids: list[str]) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]:
        if not video_ids:
            return ({}, {})

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

            async def fetch_one(bvid: str) -> tuple[str, dict | None]:
                async with semaphore:
                    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 412:
                            if not await self._validate_cookies():
                                raise AuthRequiredError("video")
                            resp = await client.get(url)
                            if resp.status_code != 200:
                                return (bvid, None)
                        elif resp.status_code != 200:
                            return (bvid, None)
                        json_data = resp.json()
                    except httpx.HTTPError:
                        return (bvid, None)
                    except AuthRequiredError:
                        raise

                    data = json_data.get("data")
                    if not isinstance(data, dict):
                        return (bvid, None)

                    return (bvid, data)

            raw_results = await asyncio.gather(*[fetch_one(bvid) for bvid in unique_bvids])

        result: dict[str, VideoMeta] = {}
        expanded: dict[str, list[str]] = {}

        for bvid, data in raw_results:
            if data is None:
                continue

            pages = data.get("pages") or []
            base_title = data.get("title", "Untitled") or "Untitled"
            base_url = f"https://www.bilibili.com/video/{bvid}"
            cover_url = data.get("pic", "") or ""
            uploader = data.get("owner", {}).get("name", "") or ""

            orig_ids = unique_bvids[bvid]
            already_has_parts = any(_split_video_id(vid)[1] is not None for vid in orig_ids)

            if len(pages) > 1:
                # Per-part metadata keyed by f"{bvid}_p{page}", used both to expand a
                # bare BVID and to fill parts that were requested individually (already
                # carry a _pN suffix) — each part keeps its own title/duration/url.
                page_meta = {}
                for page in pages:
                    p_num = page.get("page", 1)
                    part_id = f"{bvid}_p{p_num}"
                    part_name = page.get("part") or ""
                    page_meta[part_id] = VideoMeta(
                        video_id=part_id,
                        # Composite "<part> - <video>": part name leads so it survives
                        # single-line truncation (the parent video title is long and
                        # identical across parts), while the parent title still trails
                        # as context — needed when a multi-part video sits in a collection.
                        title=f"{part_name} - {base_title}" if part_name else base_title,
                        platform="bilibili",
                        source_url=f"{base_url}?p={p_num}",
                        cover_url=cover_url,
                        duration_seconds=int(page.get("duration", 0) or 0),
                        uploader=uploader,
                        part_label=f"P{p_num}",
                    )
                if already_has_parts:
                    result.update({oid: page_meta[oid] for oid in orig_ids if oid in page_meta})
                else:
                    result.update(page_meta)
                    part_ids = list(page_meta)
                    for orig_id in orig_ids:
                        expanded[orig_id] = part_ids
            else:
                meta = VideoMeta(
                    video_id=bvid,
                    title=base_title,
                    platform="bilibili",
                    source_url=data.get("short_link_v1", base_url),
                    cover_url=cover_url,
                    duration_seconds=int(data.get("duration", 0) or 0),
                    uploader=uploader,
                )
                for orig_id in orig_ids:
                    result[orig_id] = meta

        return result, expanded
