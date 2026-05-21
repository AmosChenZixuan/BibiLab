"""Query rewriter — first LLM stage in chat retrieval.

Owns the retrieve decision (retrieve true/false), query extraction, mode,
and facet detection (sequence_number / season_number). Runs over an
asymmetric context that excludes all assistant history. See issue #334.
"""

import logging
from typing import Literal

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class RewriterIntent(BaseModel):
    retrieve: bool
    query: str | None = None
    mode: Literal["narrow", "survey"] | None = None
    sequence_number: int | None = None
    season_number: int | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "RewriterIntent":
        if not self.retrieve:
            if any(v is not None for v in (self.query, self.mode, self.sequence_number, self.season_number)):
                raise ValueError("retrieve=false requires all other fields null")
        else:
            if self.query is None or self.mode is None:
                raise ValueError("retrieve=true requires query and mode")
        return self
