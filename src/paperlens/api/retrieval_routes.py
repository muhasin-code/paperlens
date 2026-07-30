"""FastAPI route handlers for retrieval debug endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.paperlens.api.schemas import Bm25SearchRequest
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.retrieval.bm25 import BM25Result, BM25Retriever
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import get_settings

logger = logging.getLogger("paperlens.api")

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

# Lazy-loaded singleton to avoid import-time side effects and repeated disk I/O
_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """Return a lazily-loaded BM25Retriever singleton."""
    global _bm25_retriever
    settings = get_settings()

    # Check if we need to (re)load: first time, or file was deleted
    if _bm25_retriever is None or not settings.bm25_index_path.exists():
        if not settings.bm25_index_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"BM25 index not found at {settings.bm25_index_path}. Run 'make bm25-build' first.",
            )
        _bm25_retriever = BM25Retriever.load(settings.bm25_index_path)
        logger.info(
            "BM25 index loaded: %d chunks, vocab=%d",
            len(_bm25_retriever.chunks),
            len({t for tokens in _bm25_retriever._tokenized_corpus for t in tokens}),
        )
    return _bm25_retriever


@router.post(
    "/bm25",
    response_model=list[BM25Result],
    status_code=status.HTTP_200_OK,
    summary="BM25 keyword search (debug endpoint)",
    description="Search the BM25 keyword index independently of the RAG pipeline. Returns top-k chunks ranked by BM25 score.",
)
async def bm25_search_endpoint(request: Bm25SearchRequest) -> list[BM25Result]:
    """Handle POST /retrieval/bm25 — standalone BM25 debug search."""
    retriever = get_bm25_retriever()
    try:
        results = retriever.search(query=request.query, top_k=request.top_k)
    except RuntimeError as exc:
        logger.exception("BM25 search failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return results


@router.post(
    "/hybrid",
    response_model=list[RetrievalResult],
    status_code=status.HTTP_200_OK,
    summary="Hybrid retrieval (semantic + BM25 with RRF)",
    description="Run semantic and BM25 retrieval in parallel, merge results with Reciprocal Rank Fusion. Return top-k chunks by RRF score.",
)
async def hybrid_search_endpoint(
    request: Request, body: Bm25SearchRequest
) -> list[RetrievalResult]:
    """Handle POST /retrieval/hybrid — hybrid retrieval debug search."""
    from src.paperlens.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    embedding_model = getattr(request.app.state, "embedding_model", None)
    bm25_retriever = getattr(request.app.state, "bm25_retriever", None)

    retriever = HybridRetriever(
        settings=settings,
        embedder=embedding_model,
        bm25_retriever=bm25_retriever,
    )
    try:
        results = retriever.search(query=body.query, top_k=body.top_k)
    except RuntimeError as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return results


@router.post(
    "/rerank",
    response_model=list[RetrievalResult],
    status_code=status.HTTP_200_OK,
    summary="Cross-encoder reranking (debug endpoint)",
    description="Rerank hybrid retrieval results using cross-encoder. Returns top-k chunks ranked by cross-encoder score.",
)
async def rerank_endpoint(request: Request, body: Bm25SearchRequest) -> list[RetrievalResult]:
    """Handle POST /retrieval/rerank — rerank hybrid retrieval results."""
    from src.paperlens.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    embedding_model = getattr(request.app.state, "embedding_model", None)
    bm25_retriever = getattr(request.app.state, "bm25_retriever", None)

    hybrid = HybridRetriever(
        settings=settings,
        embedder=embedding_model,
        bm25_retriever=bm25_retriever,
    )
    candidates = hybrid.search(query=body.query, top_k=settings.hybrid_candidate_pool)

    reranker = CrossEncoderReranker(settings)
    results = reranker.rerank(query=body.query, results=candidates, top_k=body.top_k)
    return results
