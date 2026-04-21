import json
from datetime import datetime

from pydantic import BaseModel, Field


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


def _parse_json(value: str | None) -> dict | None:
    if value:
        return json.loads(value)
    return None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "MessageResponse":
        return cls(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            metadata=_parse_json(row["metadata"]),
            created_at=_parse_datetime(row["created_at"]),
        )


class ConversationResponse(BaseModel):
    id: str
    list_id: str
    summary: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "ConversationResponse":
        return cls(
            id=row["id"],
            list_id=row["list_id"],
            summary=row["summary"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    source_ids: list[str] | None = None


class GetConversationResponse(BaseModel):
    conversation: ConversationResponse | None
    messages: list[MessageResponse]
