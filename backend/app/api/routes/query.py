"""
مسارات الـ API: GET /health و POST /query
"""
from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieval_service
from app.utils.logging_config import logger

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok" if retrieval_service.is_ready() else "vector_store_not_loaded",
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not retrieval_service.is_ready():
        raise HTTPException(status_code=503, detail="Vector store not loaded yet.")

    try:
        chunks = retrieval_service.retrieve(request.question)
        answer = generate_answer(request.question, chunks)
        sources = sorted({c["source"] for c in chunks})
        return QueryResponse(answer=answer, sources=sources)
    except RuntimeError as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
