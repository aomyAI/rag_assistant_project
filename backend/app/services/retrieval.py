"""
خدمة الاسترجاع (Retrieval): تحميل الـ Vector Store مرة واحدة عند بدء التشغيل،
واسترجاع أقرب القطع النصية لأي سؤال.
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb

from app.core.config import settings
from app.services.embeddings import get_embedding_function
from app.utils.logging_config import logger


class RetrievalService:
    """Wrapper حول Chroma collection، يُهيَّأ مرة واحدة فقط (Singleton عبر lifespan)."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedding_fn = None

    def load(self) -> None:
        logger.info("Loading vector store from %s", settings.vector_store_path)
        pipeline_config_path = Path(settings.vector_store_path) / "pipeline_config.json"
        offline_demo = settings.offline_demo_embeddings
        if pipeline_config_path.exists():
            pipeline_config = json.loads(pipeline_config_path.read_text(encoding="utf-8"))
            built_embedding_model = pipeline_config.get("embedding_model")
            if built_embedding_model and built_embedding_model != settings.embedding_model:
                raise RuntimeError(
                    "Vector store embedding model does not match EMBEDDING_MODEL: "
                    f"{built_embedding_model} != {settings.embedding_model}"
                )
            if pipeline_config.get("offline_demo_used_when_built"):
                offline_demo = True
                logger.warning(
                    "Vector store was built with offline demo embeddings; "
                    "using the matching fallback embedding function."
                )
        self._embedding_fn = get_embedding_function(
            model_name=settings.embedding_model,
            offline_demo=offline_demo,
        )
        self._client = chromadb.PersistentClient(path=settings.vector_store_path)
        self._collection = self._client.get_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            "Vector store loaded: collection='%s', count=%s",
            settings.collection_name,
            self._collection.count(),
        )

    def is_ready(self) -> bool:
        return self._collection is not None

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        if self._collection is None:
            raise RuntimeError("Vector store not loaded yet — call load() at startup.")

        k = top_k or settings.top_k
        results = self._collection.query(query_texts=[question], n_results=k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_text, meta, chunk_id, dist in zip(docs, metas, ids, distances):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "source": (meta or {}).get("source", "unknown"),
                    "distance": dist,
                }
            )
        return chunks


# نسخة واحدة تُستخدم في كل التطبيق، تُحمَّل عند الإقلاع عبر lifespan في main.py
retrieval_service = RetrievalService()
