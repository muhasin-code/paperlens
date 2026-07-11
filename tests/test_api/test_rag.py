"""Tests for the RAGService (service layer)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.paperlens.api.rag import RAGService
from src.paperlens.api.schemas import QueryRequest
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_persist_dir=tmp_path / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_model="phi4-mini",
        retrieval_top_k=5,
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
def mock_ollama_client():
    with patch("src.paperlens.api.rag.ollama.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.generate.return_value = {
            "response": "The paper proposes cosine annealing with warm restarts [2401.00001v1_chunk_0001]."
        }
        mock_client.list.return_value = {"models": [{"name": "phi4-mini"}]}
        yield mock_client


class TestRAGService:
    @pytest.mark.asyncio
    async def test_query_returns_cited_answer(self, settings, mock_retriever, mock_ollama_client):
        service = RAGService(settings, retriever=mock_retriever)
        request = QueryRequest(query="What learning rate schedule is proposed?")
        response = await service.query(request)

        assert response.answer is not None
        assert "[2401.00001v1_chunk_0001]" in response.answer
        assert len(response.citations) == 1
        cit = response.citations[0]
        assert cit.chunk_id == "2401.00001v1_chunk_0001"
        assert cit.arxiv_id == "2401.00001v1"
        assert cit.title == "Test Paper: Learning Rate Schedules"
        assert cit.authors == ["Alice Smith", "Bob Jones"]
        assert cit.section_label == "method"
        assert cit.page_start == 3
        assert cit.page_end == 4
        assert cit.score == 0.85
        assert cit.rank == 1
        assert 0.0 <= response.confidence <= 1.0
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms >= 0

    @pytest.mark.asyncio
    async def test_query_empty_retrieval_returns_fallback(
        self, settings, mock_retriever, mock_ollama_client
    ):
        mock_retriever.search.return_value = []
        service = RAGService(settings, retriever=mock_retriever)
        request = QueryRequest(query="Completely unrelated question")
        response = await service.query(request)

        assert response.answer == "I cannot answer from the provided sources."
        assert response.citations == []
        assert response.confidence == 0.0

    @pytest.mark.asyncio
    async def test_fallback_model_used_on_primary_failure(
        self, settings, mock_retriever, mock_ollama_client
    ):
        # First call fails, second succeeds
        mock_ollama_client.generate.side_effect = [
            Exception("Model not found"),
            {"response": "Fallback answer [2401.00001v1_chunk_0001]."},
        ]
        service = RAGService(settings, retriever=mock_retriever)
        request = QueryRequest(query="Test query")
        response = await service.query(request)

        assert response.answer == "Fallback answer [2401.00001v1_chunk_0001]."
        assert mock_ollama_client.generate.call_count == 2
        # Second call should use fallback model
        call_args = mock_ollama_client.generate.call_args_list[1]
        assert call_args.kwargs["model"] == "llama3.2:1b"

    @pytest.mark.asyncio
    async def test_health_check_reports_chroma_and_ollama(
        self, settings, mock_retriever, mock_ollama_client
    ):
        service = RAGService(settings, retriever=mock_retriever)
        health = await service.health_check()

        assert "chroma_vector_count" in health
        assert "ollama_reachable" in health
        assert health["ollama_reachable"] is True
