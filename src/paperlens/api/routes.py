"""FastAPI route handlers for PaperLens API."""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.paperlens.api.rag import RAGService
from src.paperlens.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.paperlens.settings import get_settings

logger = logging.getLogger("paperlens.api")

router = APIRouter(prefix="", tags=["rag"])

# Service instance created on first request (lazy) to avoid import-time side effects
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return a singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(get_settings())
    return _rag_service


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the PaperLens RAG system",
    description="Submit a natural-language question. Returns a cited answer with source chunk references.",
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Handle POST /query — main RAG endpoint."""
    service = get_rag_service()
    try:
        return await service.query(request)
    except RuntimeError as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /query")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/query/stream",
    summary="Query the PaperLens RAG system (SSE streaming)",
    description="Submit a natural-language question. Returns Server-Sent Events with token-by-token response.",
)
async def query_stream_endpoint(request: QueryRequest):
    """Handle POST /query/stream — streaming RAG endpoint."""
    service = get_rag_service()
    try:
        return StreamingResponse(
            service.query_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except RuntimeError as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /query/stream")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check with dependency status",
    description="Returns server status plus ChromaDB vector count and Ollama reachability.",
)
async def health_endpoint() -> HealthResponse:
    """Handle GET /health — extended liveness/readiness probe."""
    service = get_rag_service()
    health = await service.health_check()

    return HealthResponse(
        status="ok" if health["ollama_reachable"] else "degraded",
        chroma_collection="paperlens_chunks",
        chroma_vector_count=health["chroma_vector_count"],
        ollama_reachable=health["ollama_reachable"],
        ollama_model=service.primary_model,
        ollama_fallback_model=service.fallback_model,
    )
