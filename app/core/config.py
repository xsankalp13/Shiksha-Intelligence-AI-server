"""
Centralised settings — loaded from .env via Pydantic BaseSettings.
All other modules import from here: `from app.core.config import settings`
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # ── ERP Java Backend ──────────────────────────────────────────────────
    ERP_BASE_URL: str = "http://localhost:8080"
    ERP_API_VERSION: str = "v1"
    JWT_SECRET_KEY: str = ""
    JWT_ALLOW_MOCK: bool = True  # Set to False in production

    @property
    def erp_api_root(self) -> str:
        return f"{self.ERP_BASE_URL}/api/{self.ERP_API_VERSION}"

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_SESSION_TTL_SECONDS: int = 3600
    REDIS_AI_CONFIG_KEY: str = "ai:active_config"

    # ── LLM Provider API Keys ─────────────────────────────────────────────
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_APP_NAME: str = "Shiksha-Intelligence"

    # ── Active Model Defaults ─────────────────────────────────────────────
    ACTIVE_MODEL: str = "gemini-flash"
    INTENT_CLASSIFIER_MODEL: str = "gemini-flash"  # Always lightweight

    # ── RAG — Pinecone (Phase 2) ──────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "shiksha-knowledge"
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"      # Free plan is locked to AWS us-east-1
    RAG_TOP_K: int = 5                                # Number of chunks to retrieve per query

    # ── RAG — ChromaDB (local fallback / dev) ────────────────────────────
    CHROMA_PERSIST_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "school_knowledge"
    USE_PINECONE: bool = False                        # Set True to use Pinecone instead of Chroma

    # ── Long-term Student Memory (Optional — Supabase PostgreSQL) ─────────
    DATABASE_URL: str = ""                            # e.g. postgresql+asyncpg://user:pass@host/db
    ENABLE_LONG_TERM_MEMORY: bool = False             # Feature flag — off by default

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return upper


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call once per process."""
    return Settings()


settings: Settings = get_settings()
