"""
Shiksha AI Service — FastAPI Application Entry Point
"""
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.api.v1 import chat, ai_config
from app.services.session_service import SessionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    configure_logging()

    # ── Startup ──────────────────────────────────────────────────────────
    # Warm up Redis connection pool
    await SessionService.ping()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await SessionService.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Shiksha AI Service",
        description="Multi-model AI assistant gateway for Shiksha Intelligence ERP",
        version="1.0.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    # CORS — allow the React frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])
    app.include_router(ai_config.router, prefix="/v1", tags=["AI Config"])

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": "shiksha-ai-service"}

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
