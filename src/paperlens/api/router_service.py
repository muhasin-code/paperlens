"""Router service for unified query handling (conversational RAG vs structured extraction)."""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Literal

from pydantic import BaseModel, Field

from src.paperlens.api.rag import RAGService
from src.paperlens.api.schemas import Citation, QueryRequest
from src.paperlens.extraction.inference import ExtractionInference, get_extraction_inference
from src.paperlens.extraction.models import ExtractionOutput
from src.paperlens.extraction.router import QueryRouter, RouterDecision
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.rag")


class QueryResponse(BaseModel):
    """Unified response schema for both conversational and structured queries."""

    mode: Literal["conversational", "structured"] = Field(
        ..., description="Query routing mode used"
    )
    answer: str | None = Field(default=None, description="Generated answer (conversational path)")
    citations: list[Citation] | None = Field(
        default=None, description="Source citations (conversational path)"
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Confidence score")
    retrieval_time_ms: float | None = Field(
        default=None, description="Retrieval latency (conversational)"
    )
    generation_time_ms: float | None = Field(default=None, description="Generation latency")
    total_time_ms: float | None = Field(default=None, description="End-to-end latency")
    extraction: ExtractionOutput | None = Field(
        default=None, description="Structured extraction output"
    )
    router_decision: RouterDecision = Field(..., description="Router classification result")


class RouterService:
    """Routes queries to conversational RAG or structured extraction.

    Uses QueryRouter to classify queries and then delegates to the appropriate
    service path. Returns unified QueryResponse for both modes.
    """

    def __init__(
        self,
        settings: Settings,
        rag_service: RAGService,
        router: QueryRouter | None = None,
        extraction_inference: ExtractionInference | None = None,
    ):
        """Initialize router service.

        Args:
            settings: Application settings
            rag_service: RAG service for conversational queries
            router: Optional QueryRouter (creates new if not provided)
            extraction_inference: Optional ExtractionInference (creates new if not provided)
        """
        self.settings = settings
        self.rag_service = rag_service
        self.router = router or QueryRouter(settings)
        self.extraction_inference = extraction_inference or get_extraction_inference(settings)

    async def route_query(self, request: QueryRequest) -> QueryResponse:
        """Route a query based on classification.

        Args:
            request: The query request

        Returns:
            QueryResponse with appropriate data for the routing mode
        """
        total_start = time.perf_counter()

        decision = self.router.classify(request.query)
        logger.info(
            "Query classified as '%s' with confidence %.2f", decision.mode, decision.confidence
        )

        if decision.mode == "structured":
            return await self._handle_structured(request, decision, total_start)
        else:
            return await self._handle_conversational(request, decision, total_start)

    async def _handle_conversational(
        self,
        request: QueryRequest,
        decision: RouterDecision,
        total_start: float,
    ) -> QueryResponse:
        """Handle query through conversational RAG pipeline.

        Args:
            request: The query request
            decision: Router decision for context
            total_start: Start time for latency measurement

        Returns:
            QueryResponse with answer and citations
        """
        rag_response = await self.rag_service.query(request)

        return QueryResponse(
            mode="conversational",
            answer=rag_response.answer,
            citations=rag_response.citations,
            confidence=rag_response.confidence,
            retrieval_time_ms=rag_response.retrieval_time_ms,
            generation_time_ms=rag_response.generation_time_ms,
            total_time_ms=rag_response.total_time_ms,
            router_decision=decision,
        )

    async def _handle_structured(
        self,
        request: QueryRequest,
        decision: RouterDecision,
        total_start: float,
    ) -> QueryResponse:
        """Handle query through structured extraction pipeline.

        Args:
            request: The query request
            decision: Router decision for context
            total_start: Start time for latency measurement

        Returns:
            QueryResponse with ExtractionOutput
        """
        gen_start = time.perf_counter()

        extraction = await self.extraction_inference.extract(request.query)

        generation_time_ms = (time.perf_counter() - gen_start) * 1000
        total_time_ms = (time.perf_counter() - total_start) * 1000

        return QueryResponse(
            mode="structured",
            answer=None,
            citations=None,
            confidence=decision.confidence,
            retrieval_time_ms=0.0,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            extraction=extraction,
            router_decision=decision,
        )

    async def stream_query(
        self,
        request: QueryRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream query response (only conversational mode supports streaming).

        Args:
            request: The query request

        Yields:
            SSE-formatted response chunks
        """
        import json

        decision = self.router.classify(request.query)

        if decision.mode != "conversational":
            yield f"data: {json.dumps({'error': 'Streaming not supported for structured extraction mode only'})}\n\n"
            yield f"data: {json.dumps({'mode': 'structured', 'router_decision': decision.model_dump(), 'done': True})}\n\n"
            return

        async for chunk in self.rag_service.query_stream(request):
            yield chunk

    def get_mode(self, query: str) -> Literal["conversational", "structured"]:
        """Quick mode check without full decision object.

        Args:
            query: Query string

        Returns:
            "conversational" or "structured"
        """
        return self.router.classify(query).mode


def get_router_service(
    settings: Settings | None = None,
    rag_service: RAGService | None = None,
) -> RouterService:
    """Factory function for RouterService.

    Args:
        settings: Application settings
        rag_service: RAG service instance (creates new if not provided)

    Returns:
        RouterService instance
    """

    settings = settings or Settings()

    if rag_service is None:
        pass

        # Create a mock request context for lazy initialization
        # In production, this is called from within a request context

    return RouterService(
        settings=settings,
        rag_service=rag_service or RAGService(settings),
    )
