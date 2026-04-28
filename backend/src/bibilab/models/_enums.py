from enum import Enum
from typing import Literal


class VideoStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IN_PROGRESS = "in_progress"
    NEEDS_AUTH = "needs_auth"


ChatMode = Literal["focused", "broad"]
CHAT_MODE_FOCUSED: ChatMode = "focused"
CHAT_MODE_BROAD: ChatMode = "broad"

QueryType = Literal["factual", "breadth", "analytical"]
QUERY_TYPE_FACTUAL: QueryType = "factual"
QUERY_TYPE_BREADTH: QueryType = "breadth"
QUERY_TYPE_ANALYTICAL: QueryType = "analytical"
