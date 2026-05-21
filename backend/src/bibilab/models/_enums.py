from dataclasses import dataclass
from enum import Enum
from typing import Literal


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


Mode = Literal["narrow", "survey"]


@dataclass(frozen=True)
class RetrievalParams:
    depth_per_source: int
    top_k: int
    mode: Mode = "narrow"


# Margin values (bge logit units) indexed by mode.
# Higher margin = more aggressive filtering (keep fewer chunks).
# narrow targets a specific answer; survey lets broader matches through.
_RELEVANCE_MARGIN_BY_MODE: dict[Mode, float] = {
    "narrow": 2.0,
    "survey": 2.5,
}
