from dataclasses import dataclass
from enum import Enum


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


@dataclass(frozen=True)
class RetrievalParams:
    depth_per_source: int
    top_k: int
