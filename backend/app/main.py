"""
نقطة دخول تطبيق FastAPI: تهيئة CORS، تحميل الـ Vector Store مرة واحدة عند
الإقلاع عبر lifespan (مش في كل ريكوست)، وربط المسارات.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.utils.logging_config import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting up: loading vector store + embedding model...")
    retrieval_service.load()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RAG Document Assistant API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
