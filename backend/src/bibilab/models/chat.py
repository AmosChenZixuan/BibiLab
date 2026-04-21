import json
from datetime import datetime

from pydantic import BaseModel


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "MessageResponse":
        metadata_str = row["metadata"]
        metadata: dict | None = None
        if metadata_str:
            metadata = json.loads(metadata_str)
        created_at_str = row["created_at"]
        if isinstance(created_at_str, str):
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = created_at_str
        return cls(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            metadata=metadata,
            created_at=created_at,
        )


class ConversationResponse(BaseModel):
    id: str
    list_id: str
    summary: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "ConversationResponse":
        created_at_str = row["created_at"]
        if isinstance(created_at_str, str):
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = created_at_str
        updated_at_str = row["updated_at"]
        if isinstance(updated_at_str, str):
            updated_at = datetime.fromisoformat(updated_at_str)
        else:
            updated_at = updated_at_str
        return cls(
            id=row["id"],
            list_id=row["list_id"],
            summary=row["summary"],
            created_at=created_at,
            updated_at=updated_at,
        )


class ChatRequest(BaseModel):
    message: str


class GetConversationResponse(BaseModel):
    conversation: ConversationResponse | None
    messages: list[MessageResponse]
