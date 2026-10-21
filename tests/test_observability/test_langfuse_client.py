"""Unit tests for LangfuseClient wrapper."""

import pytest

from src.paperlens.observability.langfuse_client import (
    LangfuseClient,
    NoOpClient,
    reset_langfuse_client,
)
from src.paperlens.settings import Settings


class TestNoOpClient:
    """Tests for the NoOp client fallback."""

    def test_no_op_enabled_false(self):
        client = NoOpClient()
        assert client.enabled is False

    def test_no_op_trace_returns_self(self):
        client = NoOpClient()
        with client.trace("test") as span:
            assert span is not None
            assert span.name == "test"

    def test_no_op_span_returns_self(self):
        client = NoOpClient()
        with client.span("test_span") as span:
            assert span.name == "test_span"

    def test_no_op_flushes_without_error(self):
        client = NoOpClient()
        client.flush()  # Should not raise
        client.shutdown()  # Should not raise


class TestLangfuseClientDisabled:
    """Tests when Langfuse is disabled."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_langfuse_client()

    def test_disabled_client_returns_noop(self):
        settings = Settings(langfuse_enabled=False, langfuse_public_key="")
        client = LangfuseClient(settings)
        assert client.enabled is False
        assert isinstance(client._client, NoOpClient | type(None))

    def test_no_api_keys_returns_noop(self):
        settings = Settings(langfuse_enabled=True, langfuse_public_key="", langfuse_secret_key="")
        client = LangfuseClient(settings)
        assert client.enabled is False

    def test_context_managers_work_when_disabled(self):
        settings = Settings(langfuse_enabled=False)
        client = LangfuseClient(settings)

        with client.trace("test.trace", input="hi") as t:
            assert t is not None

        with client.span("test.span", metadata={"key": "value"}) as s:
            assert s.name == "test.span"

        client.score(name="test.score", value=0.5)  # No-op


class TestLangfuseClientScore:
    """Tests for score method."""

    def test_score_with_trace_id(self):
        settings = Settings(
            langfuse_enabled=True,
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        )
        client = LangfuseClient(settings)
        # Without real keys, this will be NoOp
        client.score(name="citation_coverage", value=1.0)  # Should not raise
