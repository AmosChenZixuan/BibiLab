from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoMeta:
    video_id: str
    title: str
    platform: str
    source_url: str
    cover_url: str
    duration_seconds: int
    uploader: str
    part_label: str | None = None

    @classmethod
    def from_source(cls, source: dict) -> "VideoMeta":
        return cls(
            video_id=source["video_id"],
            title=source["title"],
            platform=source["platform"],
            source_url=source["source_url"],
            cover_url=source["cover_url"] or "",
            duration_seconds=source["duration_seconds"],
            uploader=source["uploader"],
        )


@dataclass
class PlaylistMeta:
    playlist_id: str
    title: str
    platform: str
    source_url: str
    videos: list[VideoMeta] = field(default_factory=list)


class UnsupportedPlatformError(Exception):
    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__(f"No adapter registered for {target!r}")


class AuthRequiredError(Exception):
    def __init__(self, resource_type: str) -> None:
        self.resource_type = resource_type
        super().__init__(f"Authentication required for {resource_type!r}")


class DownloadError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class PlatformAdapter(ABC):
    @abstractmethod
    def resolve_flat(self, url: str) -> PlaylistMeta:
        """Resolve a playlist URL using extract_flat for fast enumeration without per-video metadata."""

    @abstractmethod
    async def get_videos_metadata(self, video_ids: list[str]) -> tuple[dict[str, VideoMeta], dict[str, list[str]]]:
        """Batch-fetch full metadata.
        Returns (metadata_map, expanded_map) where expanded maps original IDs to part IDs."""

    @abstractmethod
    def download(self, video_id: str, source_url: str, connections: int) -> Path:
        """Download a video and return the path to the local file.

        connections is the per-file parallel-connection budget the worker passes
        (BackendConfig.download_connections); adapters that can't use it may ignore it.
        """
