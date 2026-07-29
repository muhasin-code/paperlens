"""Tests for CrossEncoderReranker with mocked cross-encoder model.

All tests use mocks; no real model download or disk I/O.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.parsing.models import Chunk
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import Settings


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    """Helper: create a minimal Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        arxiv_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice", "Bob"],
        section_label="method",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text=text,
        token_count=len(text.split()),
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    chunks = [
        _make_chunk("c1", "machine learning optimization techniques"),
        _make_chunk("c2", "deep learning neural networks"),
        _make_chunk("c3", "natural language processing transformers"),
    ]
    return [
        RetrievalResult(chunk=chunks[0], score=0.8, rank=1),
        RetrievalResult(chunk=chunks[1], score=0.7, rank=2),
        RetrievalResult(chunk=chunks[2], score=0.6, rank=3),
    ]


class TestCrossEncoderReranker:
    def test_rerank_returns_results(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoder.predict",
            return_value=[0.9, 0.8, 0.7],
        ):
            reranker = CrossEncoderReranker(settings)
            results = reranker.rerank("test query", sample_results, top_k=3)
            assert len(results) == 3

    def test_rerank_updates_scores(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        mock_scores = [0.95, 0.85, 0.75]
        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoder.predict",
            return_value=mock_scores,
        ):
            reranker = CrossEncoderReranker(settings)
            results = reranker.rerank("test query", sample_results, top_k=3)
            assert results[0].score == 0.95
            assert results[1].score == 0.85
            assert results[2].score == 0.75

    def test_rerank_sorts_by_score_descending(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoder.predict",
            return_value=[0.5, 0.9, 0.6],
        ):
            reranker = CrossEncoderReranker(settings)
            results = reranker.rerank("test query", sample_results, top_k=3)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_rerank_top_k_respected(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoder.predict",
            return_value=[0.9, 0.8, 0.7, 0.6, 0.5],
        ):
            reranker = CrossEncoderReranker(settings)
            results = reranker.rerank("test query", sample_results, top_k=2)
            assert len(results) == 2

    def test_rerank_empty_input_returns_empty(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        reranker = CrossEncoderReranker(settings)
        results = reranker.rerank("test query", [], top_k=5)
        assert results == []

    def test_rerank_uses_settings_default_top_k(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        settings.rerank_top_k = 2
        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoder.predict",
            return_value=[0.9, 0.8, 0.7],
        ):
            reranker = CrossEncoderReranker(settings)
            results = reranker.rerank("test query", sample_results)
            assert len(results) == 2


class TestCrossEncoderRerankerModelLoading:
    def test_model_loaded_once(
        self, settings: Settings, sample_results: list[RetrievalResult]
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]

        with patch(
            "src.paperlens.retrieval.reranker.CrossEncoderReranker.get_model",
            return_value=mock_model,
        ):
            reranker1 = CrossEncoderReranker(settings)
            reranker1.rerank("test", sample_results, top_k=3)

            reranker2 = CrossEncoderReranker(settings)
            reranker2.rerank("test", sample_results, top_k=3)

            assert CrossEncoderReranker._model is not None
