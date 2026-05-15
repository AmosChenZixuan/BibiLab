from dataclasses import dataclass
from enum import Enum
from typing import Literal


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


ExpectedHits = Literal["one", "few", "many"]


@dataclass(frozen=True)
class RetrievalParams:
    depth_per_source: int
    top_k: int
    expected_hits: ExpectedHits = "few"


# Margin values (bge logit units) indexed by expected_hits.
# Higher margin = more aggressive filtering (keep fewer chunks).
# "one" targets a single answer; "many" lets broader matches through.
_RELEVANCE_MARGIN_BY_HITS: dict[ExpectedHits, float] = {
    "one": 1.0,
    "few": 2.0,
    "many": 3.0,
}
