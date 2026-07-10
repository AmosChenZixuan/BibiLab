"""TikTok platform adapter using yt-dlp. Best-effort tier: TikTok extraction
breaks in waves and the only fix is upgrading yt-dlp — generic failures carry
that hint. Public content only; login walls surface as AuthRequiredError."""

import re
from pathlib import Path

import yt_dlp

from bibilab.adapters._ytdlp_common import (
    HTTP_RETRIES,
    SOCKET_TIMEOUT,
    gather_metadata,
    pick_thumbnail,
    raise_mapped,
    safe_duration,
)
from bibilab.adapters.base import (
    PlatformAdapter,
    PlaylistMeta,
    VideoMeta,
)
from bibilab.config import downloads_dir

_AUTH_RE = re.compile(r"requiring login|log\s*in|sign\s*in|private", re.IGNORECASE)
# Photo/carousel posts have no video track; yt-dlp reports them as format-less.
_IMAGE_RE = re.compile(r"no video formats", re.IGNORECASE)

_CAPTION_LIMIT = 120
_UPGRADE_HINT = " — TikTok extraction breaks frequently; upgrading yt-dlp usually fixes it."

# Author handle is irrelevant for lookup; yt-dlp itself uses '@_' for id-only URLs.
_VIDEO_URL = "https://www.tiktok.com/@_/video/{}"


def _raise_mapped(exc: yt_dlp.utils.DownloadError) -> None:
    raise_mapped(
        exc,
        _AUTH_RE,
        message_overrides=((_IMAGE_RE, "This link is an image post, no video to transcribe."),),
        hint=_UPGRADE_HINT,
    )


def _truncate_caption(caption: str) -> str:
    """TikTok 'titles' are captions — long, hashtag-laden. Bound to 120 chars,
    cutting on a whitespace boundary when one exists in the tail half."""
    if len(caption) <= _CAPTION_LIMIT:
        return caption
    cut = caption.rfind(" ", _CAPTION_LIMIT // 2, _CAPTION_LIMIT)
    return caption[: cut if cut != -1 else _CAPTION_LIMIT].rstrip() + "…"


def _entry_to_video_meta(entry: dict) -> VideoMeta | None:
    """None when the entry carries no id — an id-less VideoMeta would corrupt dedup."""
    vid = entry.get("id")
    if not vid:
        return None
    return VideoMeta(
        video_id=vid,
        title=_truncate_caption(entry.get("title") or "Untitled"),
        platform="tiktok",
        source_url=entry.get("webpage_url") or entry.get("url") or _VIDEO_URL.format(vid),
        cover_url=pick_thumbnail(entry),
        duration_seconds=safe_duration(entry.get("duration")),
        uploader=entry.get("uploader") or entry.get("channel") or "",
    )


class TikTokAdapter(PlatformAdapter):
    def resolve_flat(self, url: str) -> PlaylistMeta:
        opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            _raise_mapped(exc)

        if info.get("_type") == "playlist":
            entries = info.get("entries") or []
            title = info.get("title", "Untitled Collection")
        else:
            entries = [info]
            title = _truncate_caption(info.get("title") or "Untitled")
        videos = [vm for e in entries if (vm := _entry_to_video_meta(e)) is not None]

        return PlaylistMeta(
            playlist_id=info.get("id", url),
            title=title,
            platform="tiktok",
            source_url=url,
            videos=videos,
        )

    async def get_videos_metadata(self, video_ids: list[str]) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]:
        def fetch_one(vid: str) -> VideoMeta | None:
            opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(_VIDEO_URL.format(vid), download=False)
            except Exception:
                # One bad video must not sink the batch — the id is simply
                # omitted; the UI keeps the flat-resolve fields as fallback.
                return None
            return _entry_to_video_meta(info)

        return (await gather_metadata(video_ids, fetch_one), {})

    def download(self, video_id: str, source_url: str, connections: int) -> Path:
        out_dir = downloads_dir()
        # TikTok files are small; the native downloader suffices — no aria2c branch.
        opts: dict = {
            "quiet": False,
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
            "format": "bestaudio/best",
            "retries": HTTP_RETRIES,
            "socket_timeout": SOCKET_TIMEOUT,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            _raise_mapped(exc)

        return out_dir / f"{video_id}.{info.get('ext', 'mp4')}"
