from dataclasses import dataclass
from enum import Enum
from typing import Literal


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


SearchMode = Literal["factual", "breadth", "analytical"]


@dataclass(frozen=True)
class RetrievalParams:
    depth_per_source: int
    top_k: int
