from pydantic import BaseModel


class ListCreateRequest(BaseModel):
    name: str


class ListResponse(BaseModel):
    id: str
    name: str
    created_at: str
