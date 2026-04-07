from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Shared error response schema.

    Use this as the detail for HTTPException in routers to ensure consistent
    error shapes across the API.
    """

    code: str
    message: str
    resource_type: str | None = None
