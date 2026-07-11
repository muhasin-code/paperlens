"""Tests for the FastAPI route handlers (API layer)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.paperlens.api.schemas import QueryResponse
from src.paperlens.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_rag_service():
    with patch("src.paperlens.api.routes.get_rag_service") as mock_get:
        mock_service = MagicMock()
        mock_service.query = AsyncMock()
        mock_service.health_check = AsyncMock()
        mock_get.return_value = mock_service
        yield mock_service


class TestQueryEndpoint:
    def test_query_success_returns_cited_answer(self, client, mock_rag_service):
        mock_rag_service.query.return_value = QueryResponse(
            answer="Test answer [2401.00001v1_chunk_0001].",
            citations=[
                {
                    "chunk_id": "2401.00001v1_chunk_0001",
                    "arxiv_id": "2401.00001v1",
                    "title": "Test Paper",
                    "authors": ["Alice"],
                    "section_label": "abstract",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.9,
                    "rank": 1,
                }
            ],
            confidence=0.9,
            retrieval_time_ms=50.0,
            generation_time_ms=1200.0,
            total_time_ms=1250.0,
        )

        resp = client.post("/query", json={"query": "What is this about?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer [2401.00001v1_chunk_0001]."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "2401.00001v1_chunk_0001"
        assert data["confidence"] == 0.9

    def test_query_validates_request_body(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422  # validation error

    def test_query_validates_top_k_bounds(self, client, mock_rag_service):
        mock_rag_service.query.return_value = QueryResponse(
            answer="x",
            citations=[],
            confidence=0.0,
            retrieval_time_ms=0,
            generation_time_ms=0,
            total_time_ms=0,
        )
        resp = client.post("/query", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422
        resp = client.post("/query", json={"query": "test", "top_k": 21})
        assert resp.status_code == 422

    def test_query_502_on_runtime_error(self, client, mock_rag_service):
        mock_rag_service.query.side_effect = RuntimeError("Both models failed")
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 502


class TestHealthEndpoint:
    def test_health_ok_when_ollama_reachable(self, client, mock_rag_service):
        mock_rag_service.health_check.return_value = {
            "chroma_vector_count": 17330,
            "ollama_reachable": True,
        }
        mock_rag_service.primary_model = "phi4-mini"
        mock_rag_service.fallback_model = "llama3.2:1b"

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chroma_collection"] == "paperlens_chunks"
        assert data["chroma_vector_count"] == 17330
        assert data["ollama_reachable"] is True
        assert data["ollama_model"] == "phi4-mini"
        assert data["ollama_fallback_model"] == "llama3.2:1b"

    def test_health_degraded_when_ollama_unreachable(self, client, mock_rag_service):
        mock_rag_service.health_check.return_value = {
            "chroma_vector_count": 17330,
            "ollama_reachable": False,
        }
        mock_rag_service.primary_model = "phi4-mini"
        mock_rag_service.fallback_model = "llama3.2:1b"

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ollama_reachable"] is False
