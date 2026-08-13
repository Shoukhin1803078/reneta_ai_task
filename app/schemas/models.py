from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
