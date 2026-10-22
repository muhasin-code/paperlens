"""Unit tests for instrumentation helpers."""

from src.paperlens.observability.instrumentation import (
    instrument_retrieval,
)


class TestInstrumentationDecoders:
    """Tests that decorators return correct wrapper types."""

    def test_instrument_retrieval_is_decorator(self):
        @instrument_retrieval
        def fake_search(self, query, top_k=None):
            return []

        import inspect

        assert inspect.isfunction(fake_search)
        assert fake_search.__name__ == "fake_search"


class TestScopeManagers:
    """Tests for trace/span/generation context managers."""

    def test_trace_context_manager_exists(self):
        from src.paperlens.observability import trace

        assert callable(trace)

    def test_span_context_manager_exists(self):
        from src.paperlens.observability import span

        assert callable(span)

    def test_generation_context_manager_exists(self):
        from src.paperlens.observability import generation

        assert callable(generation)
