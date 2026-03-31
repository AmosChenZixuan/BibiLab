from pydantic import BaseModel


class NotePathUpdateRequest(BaseModel):
    path: str | None
