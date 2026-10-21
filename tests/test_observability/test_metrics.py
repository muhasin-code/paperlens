"""Unit tests for MetricsCollector."""

import pytest

from src.paperlens.observability.metrics import MetricsCollector, Snapshot


class TestMetricsCollector:
    def test_record_latency(self):
        m = MetricsCollector()
        m.track_latency("component_a", 100.0)
        m.track_latency("component_a", 150.0)
        m.track_latency("component_a", 200.0)

        s = m.get_snapshot()
        assert "component_a" in s.latency_p50_ms
        assert s.latency_p50_ms["component_a"] == 150.0  # P50 (median) of [100, 150, 200] is 150

    def test_confidence_histogram(self):
        m = MetricsCollector()
        m.track_confidence(0.85)
        m.track_confidence(0.90)
        m.track_confidence(0.85)

        s = m.get_snapshot()
        assert s.confidence_histogram.get(0.85) == 2

    def test_citation_coverage(self):
        m = MetricsCollector()
        m.track_citation(True)
        m.track_citation(True)
        m.track_citation(False)

        s = m.get_snapshot()
        assert s.citation_coverage == pytest.approx(2 / 3, rel=0.01)

    def test_snapshot_is_dataclass(self):
        s = Snapshot()
        assert hasattr(s, "latency_p50_ms")
        assert hasattr(s, "latency_p95_ms")
        assert hasattr(s, "confidence_histogram")
        assert hasattr(s, "citation_coverage")
