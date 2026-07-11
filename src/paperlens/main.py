"""FastAPI application entry point.

Routes and dependencies are added in Phase 1 (Milestone 1.4).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.paperlens.api.routes import router as api_router
from src.paperlens.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup/shutdown logging."""
    settings = get_settings()
    import logging

    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger("paperlens")
    logger.info("PaperLens API starting on %s:%d", settings.api_host, settings.api_port)
    yield
    logger.info("PaperLens API shutting down")


app = FastAPI(
    title="PaperLens",
    description="Research intelligence over ML/AI arXiv papers.",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS for local Gradio dev (Phase 1.5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (/query, /health)
app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — basic service info."""
    return {"name": "PaperLens", "version": "0.0.1", "docs": "/docs"}
