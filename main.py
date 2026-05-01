"""
Shiksha AI Service — FastAPI Application Entry Point
"""
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.api.v1 import chat, ai_config, rag_admin, timetable
from app.services.session_service import SessionService
from app.services.rag_service import RagService
from app.services.profile_service import ProfileService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    configure_logging()

    # ── Startup ──────────────────────────────────────────────────────────
    # Warm up Redis connection pool
    await SessionService.ping()

    # Initialise RAG vector store backend (Pinecone or ChromaDB)
    await RagService.init()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await SessionService.close()
    await ProfileService.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Shiksha AI Service",
        description=(
            "Multi-model AI assistant gateway for Shiksha Intelligence ERP. "
            "Phase 2: RAG knowledge base, Entity extraction, Teacher mode, "
            "Leave filing, and Long-term student memory."
        ),
        version="2.0.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    # CORS — allow the React frontend dev server + any extra origins from settings
    _cors_origins = list(
        {
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:5173",
            *settings.EXTRA_CORS_ORIGINS,
        }
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])
    app.include_router(ai_config.router, prefix="/v1", tags=["AI Config"])
    app.include_router(rag_admin.router, prefix="/v1", tags=["RAG Admin"])
    app.include_router(timetable.router, prefix="/v1/timetable", tags=["Timetable"])

    @app.get("/ping", tags=["Health"])
    async def ping():
        """Lightweight liveness check — no external dependencies."""
        return {"status": "pong", "service": "shiksha-ai-service"}

    @app.get("/health", tags=["Health"])
    async def health():
        rag_ready = await RagService.ping()
        return {
            "status": "ok",
            "service": "shiksha-ai-service",
            "version": "2.0.0",
            "rag_ready": rag_ready,
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
