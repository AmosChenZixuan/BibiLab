import json
from datetime import datetime

from pydantic import BaseModel


class ArtifactCreateRequest(BaseModel):
    type: str
    prompt: str
    source_ids: list[str]


class ArtifactPatchRequest(BaseModel):
    name: str | None = None


class ArtifactResponse(BaseModel):
    id: str
    list_id: str
    name: str | None
    type: str
    prompt: str
    source_ids: list[str]
    status: str
    content_path: str | None
    error: str | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "ArtifactResponse":
        source_ids_raw = row["source_ids"]
        if isinstance(source_ids_raw, str):
            source_ids = json.loads(source_ids_raw) if source_ids_raw else []
        elif isinstance(source_ids_raw, list):
            source_ids = source_ids_raw
        else:
            source_ids = []
        created_at_str = row["created_at"]
        if isinstance(created_at_str, str):
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = created_at_str
        return cls(
            id=row["id"],
            list_id=row["list_id"],
            name=row["name"],
            type=row["type"],
            prompt=row["prompt"],
            source_ids=source_ids,
            status=row["status"],
            content_path=row["content_path"],
            error=row["error"],
            created_at=created_at,
        )
