"""Cross-encoder reranker for PaperLens retrieval pipeline."""

import logging
import time

from sentence_transformers import CrossEncoder

from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.observability.langfuse_client import get_langfuse_client
from src.paperlens.observability.metrics import track_latency
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.retrieval")

_observer = None


def _get_observer():
    global _observer
    if _observer is None:
        _observer = get_langfuse_client()
    return _observer


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Uses sentence-transformers.CrossEncoder to compute relevance scores
    for query-document pairs. More accurate than bi-encoder retrieval but
    slower; used to refine a small candidate pool from hybrid retrieval.
    """

    _model: CrossEncoder | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the reranker.

        Args:
            settings: Application settings (provides reranker_model, rerank_top_k).
                      If None, uses defaults.
        """
        self.settings = settings or Settings()
        self._model = None

    @classmethod
    def get_model(cls, model_name: str) -> CrossEncoder:
        """Get or create the cross-encoder model (singleton pattern).

        Args:
            model_name: HuggingFace model identifier

        Returns:
            CrossEncoder instance
        """
        if cls._model is None:
            logger.info("Loading cross-encoder model: %s", model_name)
            cls._model = CrossEncoder(model_name)
        return cls._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder.

        Args:
            query: The original query string
            results: List of RetrievalResult from hybrid retrieval
            top_k: Number of results to return (defaults to settings.rerank_top_k)

        Returns:
            List of reranked RetrievalResult objects with cross-encoder scores
        """
        if not results:
            return []

        start = time.perf_counter()
        observer = _get_observer()
        k = top_k or self.settings.rerank_top_k

        model_name = self.settings.reranker_model
        model = self.get_model(model_name)

        pairs = [(query, r.chunk.text) for r in results]
        scores: list[float] = model.predict(pairs)

        for r, score in zip(results, scores, strict=False):
            r.score = float(score)
            logger.info("Reranker score: chunk_id=%s score=%.6f", r.chunk.chunk_id, score)

        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        result = sorted_results[:k]

        elapsed = (time.perf_counter() - start) * 1000
        track_latency("reranker", elapsed)

        if observer.enabled:
            with observer.span(
                "reranker",
                metadata={
                    "reranker.input_count": len(results),
                    "reranker.output_count": len(result),
                    "reranker.latency_ms": elapsed,
                    "reranker.model": model_name,
                },
            ):
                pass

        return result
