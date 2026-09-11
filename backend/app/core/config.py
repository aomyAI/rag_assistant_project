"""
إعدادات التطبيق — تُقرأ من متغيرات البيئة (.env) عبر pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    vector_store_path: str = "./data/vector_store"
    collection_name: str = "ml_notes"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    top_k: int = 4
    frontend_origin: str = "http://localhost:8501"

    # اتركها False دائمًا عند التسليم/التشغيل الفعلي — True فقط لبيئة اختبار بدون إنترنت
    offline_demo_embeddings: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# نسخة واحدة تُستخدم في كل التطبيق (Singleton بسيط)
settings = Settings()
