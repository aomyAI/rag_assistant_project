"""
دالة الـ Embeddings المشتركة بين النوتبوك والـ backend.

الوضع الافتراضي (الموصى به دائمًا للتسليم الفعلي): نموذج حقيقي من
Sentence-Transformers، عبر SentenceTransformerEmbeddingFunction الجاهزة في chromadb.

وضع OFFLINE_DEMO: بديل بسيط بدون إنترنت (Hashing-based bag-of-words)، مفيد فقط
عند عدم توفر اتصال بالإنترنت لتحميل النموذج (مثل بيئة اختبار معزولة). لا يُستخدم
في التسليم النهائي — استخدم دائمًا الوضع الحقيقي عند التشغيل على جهازك.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from chromadb.utils import embedding_functions

EMBED_DIM = 384  # نفس أبعاد نماذج MiniLM الشائعة، للتوافق


def _hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class HashingEmbeddingFunction:
    """بديل بدائي بدون إنترنت — للتجربة/الاختبار فقط، ليس للتسليم النهائي."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "offline-hashing-fallback"


def get_embedding_function(model_name: str, offline_demo: bool = False):
    """
    يرجّع دالة الـ embeddings المناسبة.

    - offline_demo=False (الوضع الافتراضي والموصى به): نموذج Sentence-Transformers
      الحقيقي المحدد في model_name (يحتاج إنترنت في أول تشغيل لتحميل النموذج).
    - offline_demo=True: بديل بدون إنترنت لأغراض الاختبار فقط.
    """
    if offline_demo:
        return HashingEmbeddingFunction()
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
