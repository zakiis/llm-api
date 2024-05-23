from pydantic import BaseModel


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int
