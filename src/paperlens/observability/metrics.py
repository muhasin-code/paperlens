"""Metrics collection for PaperLens observability.

Provides in-process metrics collection decoupled from Langfuse:
- Latency tracking with P50/P95 percentiles per component
- Confidence score histogram (bucketed 0.0-1.0 in 0.05 increments)
- Citation coverage: ratio of queries with at least 1 cited answer
- Token usage counters per model

All methods are thread-safe and async-safe (use via event loop if needed).
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class Snapshot:
    """A point-in-time metrics snapshot."""

    latency_p50_ms: dict[str, float] = field(default_factory=dict)
    latency_p95_ms: dict[str, float] = field(default_factory=dict)
    latency_samples: int = 0
    confidence_histogram: dict[float, int] = field(default_factory=dict)
    citation_coverage: float = 0.0  # Ratio 0-1
    token_usage: dict[str, int] = field(default_factory=dict)  # model -> count
    query_count: int = 0


class MetricsCollector:
    """Collects and aggregates observability metrics in-process.

    Bounded buffers (deque with maxlen) prevent unbounded memory growth.
    Thread-safe via simple locks; for asyncio, methods are non-blocking.
    """

    MAX_SAMPLES = 1000  # Keep 1000 latency samples per component for P50/P95

    def __init__(self):
        self._latencies: dict[str, deque] = {}
        self._confidence_counts: Counter[float] = Counter()
        self._total_queries = 0
        self._cited_queries = 0
        self._token_totals: Counter[str] = Counter()
        self._lock = threading.Lock()  # Thread-safe lock

    def _ensure_component(self, name: str):
        if name not in self._latencies:
            self._latencies[name] = deque(maxlen=self.MAX_SAMPLES)

    def track_latency(self, component: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._ensure_component(component)
            self._latencies[component].append(latency_ms)

    def track_confidence(self, confidence: float | None) -> None:
        """Record a confidence score (0-1)."""
        with self._lock:
            if confidence is None:
                return
            bucket = round(confidence, 2)
            self._confidence_counts[bucket] += 1

    def track_citation(self, has_citation: bool) -> None:
        """Record whether a query returned a cited answer."""
        with self._lock:
            self._total_queries += 1
            if has_citation:
                self._cited_queries += 1

    def track_tokens(self, model: str, tokens: int) -> None:
        """Track token usage for a model."""
        with self._lock:
            self._token_totals[model] += tokens

    def get_snapshot(self) -> Snapshot:
        """Return current metrics as a Snapshot."""
        with self._lock:
            p50 = {}
            p95 = {}
            for name, values in self._latencies.items():
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                if n > 0:
                    p50[name] = round(sorted_vals[n // 2], 2)
                    p95[name] = round(sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1], 2)

            hist = dict(sorted(self._confidence_counts.items()))
            coverage = self._cited_queries / self._total_queries if self._total_queries > 0 else 0.0

            return Snapshot(
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                latency_samples=sum(len(v) for v in self._latencies.values()),
                confidence_histogram=hist,
                citation_coverage=round(coverage, 4),
                token_usage=dict(self._token_totals),
                query_count=self._total_queries,
            )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._confidence_counts.clear()
            self._token_totals.clear()
            self._total_queries = 0
            self._cited_queries = 0


# Global singleton
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def track_latency(component: str, latency_ms: float) -> None:
    get_metrics().track_latency(component, latency_ms)


def track_confidence(confidence: float | None) -> None:
    get_metrics().track_confidence(confidence)


def track_citation(has_citation: bool) -> None:
    get_metrics().track_citation(has_citation)


def track_tokens(model: str, tokens: int) -> None:
    get_metrics().track_tokens(model, tokens)


def snapshot() -> Snapshot:
    return get_metrics().get_snapshot()
