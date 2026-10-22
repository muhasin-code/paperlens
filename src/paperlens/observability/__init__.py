"""Langfuse observability module for PaperLens pipeline instrumentation.

Exports:
- LangfuseClient: wrapper with graceful degradation
- trace, span, generation, score: context managers for instrumentation
- MetricsCollector: P50/P95 latency, confidence, citation coverage tracking
- instrument_* decorators for RAG, extraction, retrieval components
"""

from src.paperlens.observability.instrumentation import (
    PipelineObserver,
    generation,
    instrument_extraction,
    instrument_generation,
    instrument_reranker,
    instrument_retrieval,
    score,
    span,
    trace,
)
from src.paperlens.observability.langfuse_client import (
    LangfuseClient,
    NoOpClient,
    get_langfuse_client,
)
from src.paperlens.observability.metrics import MetricsCollector, Snapshot, track_latency

__all__ = [
    "LangfuseClient",
    "get_langfuse_client",
    "NoOpClient",
    "trace",
    "span",
    "generation",
    "score",
    "instrument_retrieval",
    "instrument_reranker",
    "instrument_generation",
    "instrument_extraction",
    "PipelineObserver",
    "MetricsCollector",
    "Snapshot",
    "track_latency",
]
