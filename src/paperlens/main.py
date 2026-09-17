"""FastAPI application entry point.

Routes and dependencies are added in Phase 1 (Milestone 1.4).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.paperlens.api.retrieval_routes import router as retrieval_router
from src.paperlens.api.routes import router as api_router
from src.paperlens.llm import get_llm_provider
from src.paperlens.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load all heavy resources once at startup."""
    settings = get_settings()
    import logging

    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger("paperlens")
    logger.info("PaperLens API starting on %s:%d", settings.api_host, settings.api_port)

    # 1) Load LLM provider
    logger.info("Loading LLM provider (%s)...", settings.llm_provider)
    llm_provider = get_llm_provider(settings)
    app.state.llm_provider = llm_provider
    logger.info("LLM provider loaded.")

    # 2) Load embedding model into module-level cache AND app.state
    logger.info("Loading embedding model %s...", settings.embedding_model)
    from src.paperlens.api.rag import _get_embedding_model

    embedding_model = _get_embedding_model(settings)
    app.state.embedding_model = embedding_model
    logger.info(
        "Embedding model loaded (dim=%d, device=%s)",
        embedding_model.dimension,
        embedding_model.device,
    )

    # 3) Load BM25 index into module-level singleton AND app.state
    logger.info("Loading BM25 index from %s...", settings.bm25_index_path)
    from src.paperlens.api.retrieval_routes import get_bm25_retriever

    try:
        bm25_retriever = get_bm25_retriever()
        app.state.bm25_retriever = bm25_retriever
        logger.info("BM25 index loaded: %d chunks", len(bm25_retriever.chunks))
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 index not loaded at startup: %s. Run 'make bm25-build'.", exc)
        app.state.bm25_retriever = None

    # 4) Warm cross-encoder class-level singleton
    logger.info("Loading cross-encoder model %s...", settings.reranker_model)
    from src.paperlens.retrieval.reranker import CrossEncoderReranker

    CrossEncoderReranker.get_model(settings.reranker_model)
    logger.info("Cross-encoder model loaded.")

    yield

    logger.info("PaperLens API shutting down")


app = FastAPI(
    title="PaperLens",
    description="Research intelligence over ML/AI arXiv papers.",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS — allow all origins in development so Vite dev server on any host/port works.
# In production behind a reverse proxy, restrict this to your domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React static assets
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

# Include API routes (/query, /health, /query/stream)
app.include_router(api_router)
app.include_router(retrieval_router)


# SPA catch-all: serve index.html for all non-API, non-static paths
@app.get("/{full_path:path}")
async def serve_react(request: Request, full_path: str) -> FileResponse:
    """Serve React app for all non-API routes (SPA fallback)."""
    if not FRONTEND_DIST.exists():
        return FileResponse(FRONTEND_DIST / "index.html")

    file_path = FRONTEND_DIST / full_path
    if full_path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    return FileResponse(FRONTEND_DIST / "index.html")
