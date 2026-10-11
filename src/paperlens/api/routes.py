"""FastAPI route handlers for PaperLens API."""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.paperlens.api.rag import RAGService
from src.paperlens.api.schemas import HealthResponse, QueryModeRequest, QueryRequest, QueryResponse
from src.paperlens.extraction.inference import get_extraction_inference
from src.paperlens.extraction.router import QueryRouter
from src.paperlens.llm.base import LLMProvider
from src.paperlens.settings import get_settings

logger = logging.getLogger("paperlens.api")

router = APIRouter(prefix="", tags=["rag"])

# Service instance created on first request (lazy) to avoid import-time side effects
_rag_service: RAGService | None = None


def get_rag_service(request: Request) -> RAGService:
    """Return a singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        llm_provider: LLMProvider = request.app.state.llm_provider
        _rag_service = RAGService(get_settings(), llm_provider=llm_provider)
    return _rag_service


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the PaperLens system (auto-routed)",
    description="Submit a natural-language question. Uses router to select between conversational RAG and structured extraction.",
)
async def query_endpoint(request: QueryModeRequest, http_request: Request) -> QueryResponse:
    """Handle POST /query — main endpoint with auto-routing."""
    settings = get_settings()

    if request.mode == "structured":
        extraction_inference = get_extraction_inference(settings)
        extraction_output = await extraction_inference.extract(request.query)

        router = QueryRouter(settings)
        decision = router.classify(request.query)

        return QueryResponse(
            mode="structured",
            answer=None,
            citations=None,
            confidence=decision.confidence,
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
            total_time_ms=0.0,
            extraction=extraction_output.model_dump(exclude_none=False),
            router_decision=decision.model_dump(),
        )
    else:
        # Use existing RAG service for conversational/auto mode
        service = get_rag_service(http_request)
        # Create a compatible QueryRequest for RAGService
        from src.paperlens.api.schemas import QueryRequest

        rag_request = QueryRequest(query=request.query, top_k=request.top_k, model=request.model)

        try:
            return await service.query(rag_request)
        except RuntimeError as exc:
            logger.exception("RAG pipeline failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error in /query")
            raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/query/stream",
    summary="Query the PaperLens RAG system (SSE streaming)",
    description="Submit a natural-language question. Returns Server-Sent Events with token-by-token response. Only works in conversational mode.",
)
async def query_stream_endpoint(request: QueryModeRequest, http_request: Request):
    """Handle POST /query/stream — streaming RAG endpoint."""
    if request.mode == "structured":
        logger.warning("Streaming not supported for structured extraction mode")
        return StreamingResponse(
            iter([b"data: {'error': 'Streaming not available for structured extraction'}\n\n"]),
            media_type="text/event-stream",
        )

    service = get_rag_service(http_request)
    try:
        rag_request = QueryRequest(query=request.query, top_k=request.top_k, model=request.model)
        return StreamingResponse(
            service.query_stream(rag_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except RuntimeError as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /query/stream")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/extract",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Structured extraction from paper chunk",
    description="Extract structured information (research_question, method, datasets, metrics, key_finding) from the query text.",
)
async def extract_endpoint(request: QueryModeRequest, http_request: Request) -> QueryResponse:
    """Handle POST /extract — direct structured extraction endpoint."""
    settings = get_settings()

    try:
        extraction_inference = get_extraction_inference(settings)
        extraction_output = await extraction_inference.extract(request.query)

        router = QueryRouter(settings)
        decision = router.classify(request.query)

        return QueryResponse(
            mode="structured",
            answer=None,
            citations=None,
            confidence=decision.confidence,
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
            total_time_ms=0.0,
            extraction=extraction_output.model_dump(exclude_none=False),
            router_decision=decision.model_dump(),
        )
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check with dependency status",
    description="Returns server status plus ChromaDB vector count and LLM provider reachability.",
)
async def health_endpoint(http_request: Request) -> HealthResponse:
    """Handle GET /health — extended liveness/readiness probe."""
    service = get_rag_service(http_request)
    health = await service.health_check()

    settings = get_settings()
    return HealthResponse(
        status="ok" if health["llm_reachable"] else "degraded",
        chroma_collection="paperlens_chunks",
        chroma_vector_count=health["chroma_vector_count"],
        llm_reachable=health["llm_reachable"],
        llm_provider=settings.llm_provider,
        llm_model=service.primary_model,
        llm_fallback_model=service.fallback_model,
    )
