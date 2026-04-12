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
        if isinstance(source, dict):
            part_label = source.get("part_label")
        else:
            part_label = source["part_label"] if "part_label" in source.keys() else None
        return cls(
            video_id=source["video_id"],
            title=source["title"],
            platform=source["platform"],
            source_url=source["source_url"],
            cover_url=source["cover_url"] or "",
            duration_seconds=source["duration_seconds"],
            uploader=source["uploader"],
            part_label=part_label,
        )


@dataclass
class PlaylistMeta:
    playlist_id: str
    title: str
    platform: str
    source_url: str
    videos: list[VideoMeta] = field(default_factory=list)


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
    def resolve(self, url: str) -> VideoMeta | PlaylistMeta:
        """Resolve a URL to video or playlist metadata without downloading."""

    @abstractmethod
    def download(self, video_id: str, source_url: str) -> Path:
        """Download a video and return the path to the local file."""

    @abstractmethod
    def requires_auth(self, resource_type: str) -> bool:
        """Return True if the given resource type requires authentication."""
