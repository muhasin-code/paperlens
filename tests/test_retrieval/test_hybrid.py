"""Tests for HybridRetriever with RRF fusion.

All tests use mocks; no real disk I/O or network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.parsing.models import Chunk

# Import BM25Result for the tests
from src.paperlens.retrieval.bm25 import BM25Result
from src.paperlens.retrieval.hybrid import HybridRetriever
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
    return Settings(
        rrf_k=60,
        hybrid_candidate_pool=20,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        _make_chunk("c1", "machine learning optimization gradient descent"),
        _make_chunk("c2", "deep learning neural networks backpropagation"),
        _make_chunk("c3", "natural language processing transformers attention"),
        _make_chunk("c4", "computer vision image classification CNN"),
        _make_chunk("c5", "reinforcement learning policy gradients"),
    ]


@pytest.fixture
def mock_semantic_retriever(sample_chunks: list[Chunk]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = [
        RetrievalResult(chunk=sample_chunks[0], score=0.9, rank=1),
        RetrievalResult(chunk=sample_chunks[1], score=0.8, rank=2),
        RetrievalResult(chunk=sample_chunks[2], score=0.7, rank=3),
    ]
    return retriever


@pytest.fixture
def mock_bm25_retriever(sample_chunks: list[Chunk]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = [
        BM25Result(
            chunk_id="c1",
            score=1.5,
            rank=1,
            arxiv_id="2401.00001v1",
            title="Test Paper",
            authors=["Alice", "Bob"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="machine learning optimization gradient descent",
        ),
        BM25Result(
            chunk_id="c3",
            score=1.2,
            rank=1,
            arxiv_id="2401.00001v1",
            title="Test Paper",
            authors=["Alice", "Bob"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="natural language processing transformers attention",
        ),
        BM25Result(
            chunk_id="c4",
            score=1.0,
            rank=1,
            arxiv_id="2401.00001v1",
            title="Test Paper",
            authors=["Alice", "Bob"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="computer vision image classification CNN",
        ),
        BM25Result(
            chunk_id="c5",
            score=0.8,
            rank=1,
            arxiv_id="2401.00001v1",
            title="Test Paper",
            authors=["Alice", "Bob"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="reinforcement learning policy gradients",
        ),
    ]
    return retriever


class TestHybridRetriever:
    def test_search_returns_results(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,
                bm25_retriever=mock_bm25_retriever,
            )
            results = retriever.search("test", top_k=5)
            assert len(results) == 5

    def test_rrf_score_computed_correctly(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,
                bm25_retriever=mock_bm25_retriever,
            )
            results = retriever.search("test", top_k=5)
            for r in results:
                assert r.score > 0

    def test_results_sorted_by_rrf_score_descending(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,
                bm25_retriever=mock_bm25_retriever,
            )
            results = retriever.search("test", top_k=5)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,
                bm25_retriever=mock_bm25_retriever,
            )
            results = retriever.search("test", top_k=3)
            assert len(results) == 3

    def test_bm25_only_chunk_gets_rrf_score(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
    ) -> None:
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            BM25Result(
                chunk_id="c5",
                score=1.0,
                rank=1,
                arxiv_id="2401.00001v1",
                title="Test Paper",
                authors=["Alice", "Bob"],
                section_label="method",
                page_start=1,
                page_end=1,
                text="reinforcement learning policy gradients",
            ),
        ]
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,  # fixture param
                bm25_retriever=mock_bm25,  # local mock
            )
            results = retriever.search("test", top_k=5)
            chunk_ids = [r.chunk.chunk_id for r in results]
            assert "c5" in chunk_ids

    def test_rrf_score_formula_correct(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        chunk1 = _make_chunk("c1", "test content")
        mock_semantic_retriever.search.return_value = [
            RetrievalResult(chunk=chunk1, score=0.9, rank=1),
        ]
        mock_bm25_retriever.search.return_value = [
            BM25Result(
                chunk_id="c1",
                score=1.0,
                rank=1,
                arxiv_id="2401.00001v1",
                title="Test",
                authors=["A"],
                section_label="method",
                page_start=1,
                page_end=1,
                text="test content",
            ),
        ]
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,
                bm25_retriever=mock_bm25_retriever,
            )
            results = retriever.search("test", top_k=1)
            expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
            assert abs(results[0].score - expected_score) < 0.0001

    def test_empty_semantic_results_handled(
        self,
        settings: Settings,
        mock_bm25_retriever: MagicMock,
    ) -> None:
        mock_semantic = MagicMock()
        mock_semantic.search.return_value = []
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25_retriever,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic,  # local mock
                bm25_retriever=mock_bm25_retriever,  # fixture param
            )
            results = retriever.search("test", top_k=5)
            assert len(results) > 0

    def test_empty_bm25_results_handled(
        self,
        settings: Settings,
        mock_semantic_retriever: MagicMock,
    ) -> None:
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []
        with (
            patch(
                "src.paperlens.retrieval.hybrid.SemanticRetriever",
                return_value=mock_semantic_retriever,
            ),
            patch(
                "src.paperlens.retrieval.hybrid.BM25Retriever.load",
                return_value=mock_bm25,
            ),
        ):
            retriever = HybridRetriever(
                settings,
                semantic_retriever=mock_semantic_retriever,  # fixture param
                bm25_retriever=mock_bm25,  # local mock
            )
            results = retriever.search("test", top_k=5)
            assert len(results) > 0
