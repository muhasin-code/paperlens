"""Pipeline instrumentation decorators and context managers.

These helpers wrap key pipeline steps (retrieval, reranking, generation, extraction)
with Langfuse spans. They are designed to be used as:

1. Context managers inside `with` blocks for manual timing (async-compatible).
2. Decorators for automatic span entry/exit around functions.

Example:
    from src.paperlens.observability import instrument_retrieval

    @instrument_retrieval
    async def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Retrieval happens here
        return results
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.paperlens.settings import Settings

if TYPE_CHECKING:
    from src.paperlens.embedding.models import RetrievalResult

logger = logging.getLogger("paperlens.observability")

# Lazy-load to avoid import-time Langfuse dependency
_observer = None


def _get_observer() -> Any:
    global _observer
    if _observer is None:
        from src.paperlens.observability.langfuse_client import get_langfuse_client

        _observer = get_langfuse_client()
    return _observer


@contextmanager
def trace(name: str, **attrs) -> Iterator[None]:
    """Create a Langfuse trace (root span)."""
    client = _get_observer()
    with client.trace(name, **attrs) as t:
        yield t


@contextmanager
def span(name: str, **attrs) -> Iterator[Any]:
    """Create a Langfuse span."""
    client = _get_observer()
    with client.span(name, **attrs) as s:
        yield s


@contextmanager
def generation(name: str, model: str, **attrs) -> Iterator[Any]:
    """Create a Langfuse generation span for LLM calls."""
    client = _get_observer()
    with client.generation(name, model=model, **attrs) as g:
        yield g


def score(**attrs) -> None:
    """Submit a Langfuse score."""
    client = _get_observer()
    client.score(**attrs)


def instrument_retrieval(func: Callable) -> Callable:
    """Decorator: wrap a retrieval method with Langfuse span.

    Expected to decorate `HybridRetriever.search()` or `SemanticRetriever.search()`.
    The decorated function should return `list[RetrievalResult]`.

    Automatically captures: retrieval.latency_ms, retrieval.top_k, retrieval.result_count,
    retrieval.top_scores (first 3 scores), prompt_version from settings.
    """

    @functools.wraps(func)
    async def async_wrapper(self, query: str, top_k: int | None = None, *args, **kwargs):
        actual_top_k = top_k or getattr(self, "top_k", 20) or 20
        start = time.perf_counter()
        try:
            results = (
                await func(self, query, *args, top_k=top_k, **kwargs)
                if inspect.iscoroutinefunction(func)
                else func(self, query, *args, top_k=top_k, **kwargs)
            )
            elapsed = (time.perf_counter() - start) * 1000

            scores = [r.score for r in results[:3]] if results else []
            client = _get_observer()

            with client.span(
                "retrieval",
                metadata={
                    "retrieval.latency_ms": round(elapsed, 2),
                    "retrieval.top_k": actual_top_k,
                    "retrieval.result_count": len(results),
                    "retrieval.top_scores": scores,
                    "prompt_version": Settings().prompt_version,
                },
            ):
                return results
        except Exception as e:
            logger.exception("Retrieval instrumentation failed: %s", e)
            return (
                await func(self, query, *args, top_k=top_k, **kwargs)
                if inspect.iscoroutinefunction(func)
                else func(self, query, *args, top_k=top_k, **kwargs)
            )

    @functools.wraps(func)
    def sync_wrapper(self, query: str, top_k: int | None = None, *args, **kwargs):
        actual_top_k = top_k or getattr(self, "top_k", 20) or 20
        start = time.perf_counter()
        try:
            results = func(self, query, *args, top_k=top_k, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            scores = [r.score for r in results[:3]] if results else []
            client = _get_observer()

            with client.span(
                "retrieval",
                metadata={
                    "retrieval.latency_ms": round(elapsed, 2),
                    "retrieval.top_k": actual_top_k,
                    "retrieval.result_count": len(results),
                    "retrieval.top_scores": scores,
                    "prompt_version": Settings().prompt_version,
                },
            ):
                return results
        except Exception as e:
            logger.exception("Retrieval instrumentation failed: %s", e)
            return func(self, query, *args, top_k=top_k, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


def instrument_reranker(func: Callable) -> Callable:
    """Decorator: wrap CrossEncoderReranker.rerank() with span.

    Captures: reranker.input_count, reranker.output_count, reranker.latency_ms,
    reranker.model.
    """

    @functools.wraps(func)
    def wrapper(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None, *args, **kwargs
    ):
        start = time.perf_counter()
        input_count = len(results)
        try:
            ranked = func(self, query, results, *args, top_k=top_k, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            client = _get_observer()

            with client.span(
                "reranker",
                metadata={
                    "reranker.input_count": input_count,
                    "reranker.output_count": len(ranked),
                    "reranker.latency_ms": round(elapsed, 2),
                    "reranker.model": getattr(self, "settings", Settings()).reranker_model,
                },
            ):
                return ranked
        except Exception as e:
            logger.exception("Reranker instrumentation failed: %s", e)
            return func(self, query, results, *args, top_k=top_k, **kwargs)

    return wrapper


def instrument_generation(func: Callable) -> Callable:
    """Decorator: wrap LLM generation with span.

    Use this on methods that call `llm_provider.generate()` or similar.
    Captures: generation.model, generation.tokens_in, generation.tokens_out,
    generation.latency_ms.
    """

    @functools.wraps(func)
    async def async_wrapper(
        self, prompt: str, model: str, options: dict | None = None, *args, **kwargs
    ):
        start = time.perf_counter()
        try:
            result = await func(self, prompt, model, options, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            client = _get_observer()

            with client.generation(
                name="llm.generate",
                model=model,
                input=prompt,
                output=result,
                metadata={
                    "generation.latency_ms": round(elapsed, 2),
                    "generation.tokens_in": len(prompt.split()),
                    "generation.tokens_out": len(result.split()) if result else 0,
                },
            ):
                return result
        except Exception as e:
            logger.exception("Generation instrumentation failed: %s", e)
            raise

    return async_wrapper


def instrument_extraction(func: Callable) -> Callable:
    """Decorator: wrap ExtractionInference.extract() with span.

    Captures: extraction.mode, extraction.latency_ms, extraction.tokens_out,
    extraction.fields_populated.
    """

    @functools.wraps(func)
    async def async_wrapper(self, chunk_text: str, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(self, chunk_text, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            client = _get_observer()

            # Count non-null fields
            fields = ["research_question", "method", "datasets", "metrics", "key_finding"]
            populated = sum(1 for f in fields if getattr(result, f, None) is not None)

            with client.generation(
                name="extraction",
                model=self.settings.extraction_adapter_repo,
                input=chunk_text,
                output=result.model_dump() if hasattr(result, "model_dump") else str(result),
                metadata={
                    "extraction.mode": self.mode,
                    "extraction.latency_ms": round(elapsed, 2),
                    "extraction.tokens_out": len(chunk_text.split()),
                    "extraction.fields_populated": populated,
                },
            ):
                return result
        except Exception as e:
            logger.exception("Extraction instrumentation failed: %s", e)
            raise

    return async_wrapper


class PipelineObserver:
    """Convenience class for obtaining instrumented spans directly in code."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._client = _get_observer()

    @property
    def enabled(self) -> bool:
        return self._client.enabled

    @contextmanager
    def trace(self, name: str, **attrs):
        """Create a trace (root span)."""
        with self._client.trace(name, **attrs) as t:
            yield t

    @contextmanager
    def span(self, name: str, **attrs):
        """Create a span."""
        with self._client.span(name, **attrs) as s:
            yield s

    @contextmanager
    def generation(self, name: str, model: str, **attrs):
        """Create a generation span."""
        with self._client.generation(name, model=model, **attrs) as g:
            yield g

    def score(self, **attrs):
        """Submit a score."""
        self._client.score(**attrs)


def get_observer(settings: Settings | None = None) -> PipelineObserver:
    return PipelineObserver(settings)
