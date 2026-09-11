"""
نماذج الطلب والاستجابة الخاصة بـ /query.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="سؤال المستخدم")


class SourceChunk(BaseModel):
    document: str
    chunk_id: str
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
