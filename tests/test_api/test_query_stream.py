"""Tests for the streaming query endpoint (/query/stream)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.paperlens.api.rag import RAGService
from src.paperlens.api.schemas import QueryRequest
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.llm.base import LLMProvider
from src.paperlens.observability.metrics import get_metrics
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_persist_dir=tmp_path / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_model="phi4-mini",
        retrieval_top_k=5,
        llm_provider="openai_compat",
        llm_model="openai/gpt-oss-120b",
        llm_fallback_model="qwen/qwen3-8b",
        langfuse_enabled=False,
    )


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock(spec=SemanticRetriever)
    chunk = Chunk(
        chunk_id="2401.00001v1_chunk_0001",
        arxiv_id="2401.00001v1",
        title="Test Paper: Learning Rate Schedules",
        authors=["Alice Smith", "Bob Jones"],
        section_label="method",
        chunk_index=0,
        page_start=3,
        page_end=4,
        text="We propose a cosine annealing schedule with warm restarts...",
        token_count=120,
        total_chunks_in_section=5,
        total_chunks_in_paper=20,
    )
    retriever.search.return_value = [
        RetrievalResult(chunk=chunk, score=0.85, rank=1),
    ]
    return retriever


@pytest.fixture
def mock_vector_store():
    with patch("src.paperlens.api.rag.VectorStore") as mock_vs_class:
        mock_vs = MagicMock()
        mock_vs_class.return_value = mock_vs
        mock_vs.count.return_value = 17330
        yield mock_vs


def parse_sse(chunks: list[str]) -> list[dict]:
    """Parse concatenated SSE chunks into a list of event dicts."""
    events = []
    for block in "".join(chunks).split("\n\n"):
        line = block.strip()
        if not line:
            continue
        assert line.startswith("data: ")
        events.append(json.loads(line[len("data: ") :]))
    return events


def make_async_gen(*chunks: str):
    """Return an async-generator function that yields the given chunks."""

    async def gen(*_args, **_kwargs):
        for c in chunks:
            yield c

    return gen


class TestQueryStream:
    @pytest.fixture(autouse=True)
    def _reset_metrics(self):
        get_metrics().reset()
        yield
        get_metrics().reset()

    @pytest.mark.asyncio
    async def test_happy_path_events_order(self, settings, mock_retriever, mock_vector_store):
        """Citations event, then token events, then final event with done=true."""
        provider = MagicMock(spec=LLMProvider)
        provider.stream = MagicMock(
            side_effect=make_async_gen(
                "The paper proposes ",
                "cosine annealing [2401.00001v1_chunk_0001].",
            )
        )

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="What learning rate schedule is proposed?")

        chunks = [c async for c in service.query_stream(request)]
        events = parse_sse(chunks)

        # Event 0: citations + retrieval_time_ms
        assert "citations" in events[0]
        assert "retrieval_time_ms" in events[0]
        assert "token" not in events[0]
        assert len(events[0]["citations"]) == 1

        # Middle events: only token
        for ev in events[1:-1]:
            assert set(ev.keys()) == {"token"}
            assert ev["token"]

        # Final event
        final = events[-1]
        assert final["done"] is True
        assert final["ttft_ms"] >= 0
        assert final["ttft_ms"] <= final["generation_time_ms"]
        assert final["chunk_count"] == 2
        assert final["model"] == "openai/gpt-oss-120b"
        assert "refused" not in final

        # stream called once on primary
        assert provider.stream.call_count == 1
        # Verify options passed
        call_kwargs = provider.stream.call_args.kwargs
        assert call_kwargs["options"]["system"]
        assert call_kwargs["options"]["num_predict"] == 1024

    @pytest.mark.asyncio
    async def test_options_passed_to_stream(self, settings, mock_retriever, mock_vector_store):
        """system and num_predict are forwarded to provider.stream()."""
        provider = MagicMock(spec=LLMProvider)
        provider.stream = MagicMock(side_effect=make_async_gen("ok"))

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")
        _ = [c async for c in service.query_stream(request)]

        call_kwargs = provider.stream.call_args.kwargs
        assert call_kwargs["options"]["system"]
        assert call_kwargs["options"]["num_predict"] == 1024

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_single_event(
        self, settings, mock_retriever, mock_vector_store
    ):
        """Empty retrieval yields one event, does not call stream()."""
        mock_retriever.search.return_value = []

        provider = AsyncMock(spec=LLMProvider)

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Unrelated")
        chunks = [c async for c in service.query_stream(request)]
        events = parse_sse(chunks)

        assert len(events) == 1
        assert events[0]["citations"] == []
        assert events[0]["generation_time_ms"] == 0
        assert events[0]["done"] is True
        assert provider.stream.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_before_first_chunk(self, settings, mock_retriever, mock_vector_store):
        """Primary raises before any chunk -> fallback used."""
        call_count = 0

        async def failing_then_ok(prompt, model, options=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("primary failed")
            # Second call (fallback) yields one chunk
            for c in ["fallback chunk"]:
                yield c

        provider = AsyncMock(spec=LLMProvider)
        provider.stream = failing_then_ok

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")
        chunks = [c async for c in service.query_stream(request)]
        events = parse_sse(chunks)

        final = events[-1]
        assert final["model"] == "qwen/qwen3-8b"
        # One token from fallback
        token_events = [e for e in events if "token" in e]
        assert len(token_events) == 1

    @pytest.mark.asyncio
    async def test_primary_yields_then_raises_no_fallback(
        self, settings, mock_retriever, mock_vector_store
    ):
        """Primary yields a chunk then raises -> exception propagates, no fallback."""

        async def gen_then_raise(prompt, model, options=None):
            yield "first chunk"
            raise RuntimeError("stream died")

        provider = MagicMock(spec=LLMProvider)
        provider.stream = MagicMock(side_effect=gen_then_raise)

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")

        with pytest.raises(RuntimeError, match="Stream failed after 1 chunks"):
            _ = [c async for c in service.query_stream(request)]

        # stream called once, no fallback attempt
        assert provider.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_exact_refusal_clears_citations_on_final(
        self, settings, mock_retriever, mock_vector_store
    ):
        """Exact refusal message sets refused=true and citations=[] on final."""
        refusal_msg = "I cannot answer from the provided sources."

        provider = AsyncMock(spec=LLMProvider)
        provider.stream = make_async_gen(refusal_msg)

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")
        chunks = [c async for c in service.query_stream(request)]
        events = parse_sse(chunks)

        # Event 0 still had citations
        assert events[0]["citations"]
        # Final event clears them
        final = events[-1]
        assert final["refused"] is True
        assert final["citations"] == []
        assert final["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_substring_refusal_does_not_set_refused(
        self, settings, mock_retriever, mock_vector_store
    ):
        """Completion containing refusal sentence as substring is not refused."""
        provider = AsyncMock(spec=LLMProvider)
        provider.stream = make_async_gen("This is not the refusal sentence.")

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")
        chunks = [c async for c in service.query_stream(request)]
        events = parse_sse(chunks)

        final = events[-1]
        assert "refused" not in final
        assert final["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_metrics_stream_ttft_recorded(self, settings, mock_retriever, mock_vector_store):
        """track_latency('stream.ttft', ...) is called on happy path."""
        provider = AsyncMock(spec=LLMProvider)
        provider.stream = make_async_gen("hello")

        service = RAGService(settings, llm_provider=provider, retriever=mock_retriever)
        request = QueryRequest(query="Test")
        _ = [c async for c in service.query_stream(request)]

        snap = get_metrics().get_snapshot()
        assert "stream.ttft" in snap.latency_p50_ms


class TestQueryStreamEndpoint:
    """Integration tests via TestClient for the /query/stream route."""

    def test_structured_mode_returns_valid_json(self):
        """POST /query/stream with mode=structured returns parseable SSE event."""
        from fastapi.testclient import TestClient

        from src.paperlens.main import app

        client = TestClient(app)
        resp = client.post(
            "/query/stream",
            json={"query": "extract the datasets", "mode": "structured"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse the SSE body
        body = resp.text.strip()
        assert body.startswith("data: ")
        payload = json.loads(body[len("data: ") :])
        assert payload["error"] == "Streaming not available for structured extraction"
        assert payload["done"] is True

    def test_structured_mode_does_not_call_rag_service(self):
        """Structured branch returns before get_rag_service is invoked."""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from src.paperlens.main import app

        with patch("src.paperlens.api.routes.get_rag_service") as mock_get:
            client = TestClient(app)
            client.post("/query/stream", json={"query": "extract", "mode": "structured"})
            mock_get.assert_not_called()
