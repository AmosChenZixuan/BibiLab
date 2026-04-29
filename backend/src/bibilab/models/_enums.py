from enum import Enum
from typing import Literal


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


ChatMode = Literal["auto", "focused", "broad"]
CHAT_MODE_AUTO: ChatMode = "auto"
CHAT_MODE_FOCUSED: ChatMode = "focused"
CHAT_MODE_BROAD: ChatMode = "broad"

QueryType = Literal["factual", "breadth", "analytical"]
QUERY_TYPE_FACTUAL: QueryType = "factual"
QUERY_TYPE_BREADTH: QueryType = "breadth"
QUERY_TYPE_ANALYTICAL: QueryType = "analytical"


def map_type_to_mode(qt: QueryType) -> ChatMode:
    if qt == QUERY_TYPE_FACTUAL:
        return CHAT_MODE_FOCUSED
    if qt in (QUERY_TYPE_BREADTH, QUERY_TYPE_ANALYTICAL):
        return CHAT_MODE_BROAD
    raise ValueError(f"Unknown query type: {qt!r}")
