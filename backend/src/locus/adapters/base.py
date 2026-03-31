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
