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
