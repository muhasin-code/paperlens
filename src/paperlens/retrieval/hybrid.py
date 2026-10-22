"""Hybrid retrieval: semantic + BM25 with Reciprocal Rank Fusion."""

import time
from collections import defaultdict

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.observability.langfuse_client import get_langfuse_client
from src.paperlens.parsing.models import Chunk
from src.paperlens.retrieval.bm25 import BM25Retriever
from src.paperlens.settings import Settings

_observer = None


def _get_observer():
    global _observer
    if _observer is None:
        _observer = get_langfuse_client()
    return _observer


class HybridRetriever:
    """Combines semantic and BM25 retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        settings: Settings,
        semantic_retriever: SemanticRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            settings: Application settings (provides rrf_k, hybrid_candidate_pool)
            semantic_retriever: Optional pre-built SemanticRetriever (for testing)
            bm25_retriever: Optional pre-built BM25Retriever (for testing)
        """
        self.settings = settings
        self._semantic_retriever = semantic_retriever
        self._bm25_retriever = bm25_retriever
        self._embedder = embedder
        self._rrf_k = settings.rrf_k
        self._candidate_pool = settings.hybrid_candidate_pool
        # Weighted RRF: semantic gets 2x weight vs BM25
        self._semantic_weight = 2.0
        self._bm25_weight = 1.0

    def _get_semantic_retriever(self) -> SemanticRetriever:
        """Lazily create SemanticRetriever if not provided."""
        if self._semantic_retriever is None:
            self._semantic_retriever = SemanticRetriever(self.settings, embedder=self._embedder)
        return self._semantic_retriever

    def _get_bm25_retriever(self) -> BM25Retriever:
        """Lazily create BM25Retriever if not provided."""
        if self._bm25_retriever is None:
            if not self.settings.bm25_index_path.exists():
                raise RuntimeError(
                    f"BM25 index not found at {self.settings.bm25_index_path}. "
                    "Run 'make bm25-build' first."
                )
            self._bm25_retriever = BM25Retriever.load(self.settings.bm25_index_path)
        return self._bm25_retriever

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Execute hybrid retrieval with RRF fusion.

        Runs both semantic and BM25 retrieval, then merges results using
        weighted Reciprocal Rank Fusion (semantic: 2.0, BM25: 1.0).

        Args:
            query: Natural language query string
            top_k: Number of final results to return (defaults to settings)

        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        start = time.perf_counter()
        observer = _get_observer()
        k = top_k or self.settings.retrieval_top_k
        pool_size = self._candidate_pool

        semantic = self._get_semantic_retriever()
        bm25 = self._get_bm25_retriever()

        semantic_start = time.perf_counter()
        semantic_results = semantic.search(query, top_k=pool_size)
        semantic_ms = (time.perf_counter() - semantic_start) * 1000

        bm25_start = time.perf_counter()
        bm25_results = bm25.search(query, top_k=pool_size)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        # For semantic results (RetrievalResult has .chunk) - weighted 2.0
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.chunk.chunk_id
            rrf_scores[chunk_id] += self._semantic_weight / (self._rrf_k + rank)
            chunk_map[chunk_id] = result.chunk

        # For BM25 results (BM25Result has fields directly, no .chunk) - weighted 1.0
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk_id  # Direct access
            rrf_scores[chunk_id] += self._bm25_weight / (self._rrf_k + rank)
            if chunk_id not in chunk_map:
                # Reconstruct Chunk from BM25Result fields
                chunk_map[chunk_id] = Chunk(
                    chunk_id=result.chunk_id,
                    arxiv_id=result.arxiv_id,
                    title=result.title,
                    authors=result.authors,
                    section_label=result.section_label,
                    chunk_index=0,  # Not stored in BM25Result
                    page_start=result.page_start,
                    page_end=result.page_end,
                    text=result.text,
                    token_count=len(result.text.split()),
                    total_chunks_in_section=0,
                    total_chunks_in_paper=0,
                )

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        final_start = time.perf_counter()
        for rank, (chunk_id, score) in enumerate(sorted_chunks[:k], start=1):
            chunk = chunk_map[chunk_id]
            final_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )
        final_ms = (time.perf_counter() - final_start) * 1000
        total_ms = (time.perf_counter() - start) * 1000

        with observer.span(
            "retrieval.hybrid",
            metadata={
                "retrieval.semantic_ms": semantic_ms,
                "retrieval.bm25_ms": bm25_ms,
                "retrieval.rrf_ms": final_ms,
                "retrieval.total_ms": total_ms,
                "hybrid.semantic.results": len(semantic_results),
                "hybrid.bm25.results": len(bm25_results),
                "hybrid.rrf.fused_count": len(final_results),
                "rrf_k": self._rrf_k,
            },
        ):
            return final_results

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(rrf_k={self._rrf_k}, "
            f"candidate_pool={self._candidate_pool}, "
            f"semantic_weight={self._semantic_weight}, "
            f"bm25_weight={self._bm25_weight}, "
            f"semantic_retriever={'set' if self._semantic_retriever else 'lazy'}, "
            f"bm25_retriever={'set' if self._bm25_retriever else 'lazy'})"
        )
