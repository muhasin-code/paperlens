"""Langfuse client wrapper with graceful degradation.

Wraps the official Langfuse SDK (v4.x) so the rest of PaperLens never imports
`langfuse` directly. This provides two invariants:

1. Never raise: every SDK call is defensive. If Langfuse is disabled, missing
   keys, unimportable, or unreachable, the wrapper degrades to a no-op client.
2. Never block: the SDK flushes asynchronously. A network failure logs a warning
   but never blocks the request path.

Usage:
    from src.paperlens.observability import LangfuseClient, get_langfuse_client

    client = get_langfuse_client()  # singleton, lazy init
    with client.trace("rag.query", input=query, user_id=user_id) as trace:
        # all child spans automatically propagate within this task
        with client.span("retrieval.semantic", latency_ms=..., top_k=...):
            ...
        with client.generation(name="llm.generate", model="phi4-mini", input=..., output=..., tokens_in=..., tokens_out=...):
            ...
        client.score(name="citation_coverage", value=1.0, comment="all answers cited")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from langfuse.types import TraceContext
except ImportError:
    TraceContext = None  # type: ignore[assignment,misc]

from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.observability")

# Global client singleton
_client: LangfuseClient | None = None


class NoOpClient:
    """No-op client returned when Langfuse is disabled or unavailable."""

    enabled = False

    @contextmanager
    def trace(self, name: str, **kwargs) -> Iterator[None]:
        """No-op trace context manager."""
        yield _NoOpSpan(name=name)

    @contextmanager
    def span(self, name: str, **kwargs) -> Iterator[None]:
        """No-op span context manager."""
        yield _NoOpSpan(name=name)

    @contextmanager
    def generation(self, name: str, **kwargs) -> Iterator[None]:
        """No-op generation context manager."""
        yield _NoOpSpan(name=name)

    def score(self, *, name: str, value: float, **kwargs) -> None:
        """No-op score submission."""
        return None

    def flush(self) -> None:
        """No-op flush."""
        return None

    def shutdown(self) -> None:
        """No-op shutdown."""
        return None


class LangfuseClient:
    """Wrapper around Langfuse SDK v4.x with graceful degradation.

    Provides helper methods `trace()`, `span()`, `generation()`, `score()` that
    work seamlessly with both the SDK and a no-op fallback.

    OTel Note (langfuse v4.x):
    The SDK uses OpenTelemetry's ContextVars for span propagation. Within a
    single async task, child spans created inside a `with` block automatically
    attach to the parent. Cross-task propagation (asyncio.gather) would need
    explicit `ContextVar.set()` – out of scope for this milestone.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._client = self._init_client()
        self._enabled = self._client is not None

    def _init_client(self) -> Any | None:
        """Initialize Langfuse client or return None if disabled/unavailable."""
        if not self.settings.langfuse_enabled:
            logger.debug("Langfuse disabled via langfuse_enabled=False")
            return None

        public_key = self.settings.langfuse_public_key
        secret_key = self.settings.langfuse_secret_key

        if not public_key or not secret_key:
            logger.warning("Langfuse keys not set; using UI signup flow")
            # Let SDK handle initialization (will use defaults, user can set keys later)
            # Or bail out to no-op if we require explicit keys
            return None

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=self.settings.langfuse_host,
            )
            logger.info("Langfuse client initialized (host=%s)", self.settings.langfuse_host)
            return client
        except ImportError as e:
            logger.warning("Langfuse SDK not installed: %s", e)
            return None
        except Exception as e:
            logger.warning("Failed to initialize Langfuse client: %s", e)
            return None

    @property
    def enabled(self) -> bool:
        """Whether tracing is active."""
        return self._enabled and self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Create a trace (root span) with Langfuse.

        The trace captures the overall request/response. Child spans created
        via `span()`, `generation()` automatically nest within this trace.
        """
        if not self.enabled:
            yield _NoOpSpan(name)
            return

        trace_id = self._client.create_trace_id()
        trace_ctx = TraceContext(trace_id=trace_id)

        # Merge user_id, session_id, tags into metadata
        merged_metadata = {**(metadata or {})}
        if user_id is not None:
            merged_metadata["user_id"] = user_id
        if session_id is not None:
            merged_metadata["session_id"] = session_id
        if tags:
            merged_metadata["tags"] = tags

        with self._client.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=trace_ctx,
            input=input,
            output=output,
            metadata=merged_metadata,
        ) as trace:
            trace.update(metadata={"input_length": len(str(input)) if input else 0})
            yield trace

    @contextmanager
    def span(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        input: Any = None,
        output: Any = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> Iterator[Any]:
        """Create a child span within an existing trace context.

        If no trace is active, this is a no-op.
        """
        if not self.enabled:
            yield _NoOpSpan(name)
            return

        with self._client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input,
            output=output,
            metadata=metadata or {},
            level=level,
            status_message=status_message,
        ) as obs:
            yield obs

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        model_parameters: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ) -> Iterator[Any]:
        """Create a generation span for LLM calls.

        Supports OpenTelemetry duration tracking and token usage attribution.
        """
        if not self.enabled:
            yield _NoOpSpan(name)
            return

        with self._client.start_as_current_observation(
            name=name,
            as_type="generation",
            input=input,
            output=output,
            model=model,
            model_parameters=model_parameters or {},
            usage_details=usage_details or {},
            metadata=metadata or {},
            level=level,
        ) as gen:
            yield gen

    def score(
        self,
        *,
        name: str,
        value: float,
        trace_id: str | None = None,
        observation_id: str | None = None,
        comment: str | None = None,
        data_type: str = "NUMERIC",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Submit a custom score to Langfuse.

        Arguments:
            name: Score identifier (e.g., "citation_coverage", "latency_p50")
            value: Numeric score (0-1 for coverage, milliseconds for latency, etc.)
            trace_id: Optional; auto-detected from current trace if not provided
            observation_id: Optional; auto-detected for per-span scores
            comment: Human-readable note
            data_type: NUMERIC, CATEGORICAL, TEXT, etc.
            metadata: Additional key-value pairs
        """
        if not self.enabled:
            return

        try:
            self._client.create_score(
                name=name,
                value=value,
                trace_id=trace_id or self._client.get_current_trace_id(),
                observation_id=observation_id or self._client.get_current_observation_id(),
                comment=comment,
                data_type=data_type,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to submit Langfuse score '%s': %s", name, e)

    def latency_score(self, latency_ms: float, p: str = "p50") -> None:
        """Convenience method to record latency percentiles at trace level."""
        self.score(name=f"latency_{p}", value=round(latency_ms, 2), comment=f"Time to {p} latency")

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enabled and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug("Langfuse flush (best-effort): %s", e)

    def shutdown(self) -> None:
        """Gracefully shutdown the client and flush pending data."""
        if self.enabled and self._client:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as e:
                logger.debug("Langfuse shutdown (best-effort): %s", e)


class _NoOpSpan:
    """Helper returned by no-op context managers for attribute access."""

    def __init__(self, name: str):
        self.name = name
        self._id = None

    @property
    def id(self) -> str | None:
        return self._id

    def update(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass

    def score(self, **kwargs) -> None:
        pass


def get_langfuse_client(settings: Settings | None = None) -> LangfuseClient:
    """Return a singleton LangfuseClient instance."""
    global _client
    if _client is None:
        _client = LangfuseClient(settings)
    return _client


def reset_langfuse_client() -> None:
    """Reset the singleton (useful for testing)."""
    global _client
    if _client is not None:
        _client.shutdown()
    _client = None
