"""Tests for BM25Retriever and BM25Result.

All tests use in-memory Chunk objects and tmp_path for persistence.
No network or real file I/O outside tmp_path.
"""

from pathlib import Path

import pytest

from src.paperlens.parsing.models import Chunk
from src.paperlens.retrieval.bm25 import BM25Result, BM25Retriever

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chunk(chunk_id: str, text: str, section: str = "method") -> Chunk:
    """Helper: create a minimal Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        arxiv_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice", "Bob"],
        section_label=section,
        chunk_index=0,
        page_start=1,
        page_end=1,
        text=text,
        token_count=len(text.split()),
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Five chunks with distinct vocabulary for BM25 testing."""
    return [
        _make_chunk("c1", "machine learning gradient descent optimization"),
        _make_chunk("c2", "deep learning neural networks backpropagation"),
        _make_chunk("c3", "machine learning regularization dropout overfitting"),
        _make_chunk("c4", "natural language processing transformers attention"),
        _make_chunk("c5", "machine learning reinforcement learning policy gradients"),
    ]


@pytest.fixture
def retriever(tmp_path: Path, sample_chunks: list[Chunk]) -> BM25Retriever:
    """BM25Retriever backed by sample_chunks, persisting to tmp_path."""
    index_path = tmp_path / "test_bm25.pkl"
    r = BM25Retriever(sample_chunks, index_path=index_path)
    r.build()
    r.persist()
    return r


# ── BM25Result model ─────────────────────────────────────────────────────────


class TestBM25Result:
    def test_model_creation(self) -> None:
        result = BM25Result(
            chunk_id="c1",
            score=1.5,
            rank=1,
            arxiv_id="2401.00001v1",
            title="T",
            authors=["A"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="hello world",
        )
        assert result.chunk_id == "c1"
        assert result.score == 1.5
        assert result.rank == 1

    def test_model_dump_roundtrip(self) -> None:
        result = BM25Result(
            chunk_id="c1",
            score=1.0,
            rank=1,
            arxiv_id="2401.00001v1",
            title="T",
            authors=["A"],
            section_label="method",
            page_start=1,
            page_end=1,
            text="hello",
        )
        dumped = result.model_dump()
        assert dumped["chunk_id"] == "c1"
        assert dumped["score"] == 1.0


# ── BM25Retriever initialization ─────────────────────────────────────────────


class TestBM25RetrieverInit:
    def test_default_index_path_is_class_constant(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        assert r.index_path == BM25Retriever.INDEX_PATH

    def test_custom_index_path(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        custom = tmp_path / "custom.pkl"
        r = BM25Retriever(sample_chunks, index_path=custom)
        assert r.index_path == custom

    def test_index_path_property(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        custom = tmp_path / "custom.pkl"
        r = BM25Retriever(sample_chunks, index_path=custom)
        assert r.index_path == custom


# ── BM25Retriever.build ──────────────────────────────────────────────────────


class TestBM25RetrieverBuild:
    def test_build_returns_stats(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        stats = r.build()
        assert stats["corpus_size"] == 5
        assert stats["vocab_size"] > 0
        assert stats["build_time_ms"] > 0
        assert stats["k1"] == 1.5
        assert stats["b"] == 0.75

    def test_build_sets_internal_state(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        r.build()
        assert r._bm25 is not None
        assert len(r._tokenized_corpus) == 5

    def test_build_custom_k1_b(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks, k1=2.0, b=0.5)
        stats = r.build()
        assert stats["k1"] == 2.0
        assert stats["b"] == 0.5


# ── BM25Retriever.search ─────────────────────────────────────────────────────


class TestBM25RetrieverSearch:
    def test_search_returns_results(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=3)
        assert len(results) == 3

    def test_search_results_have_correct_type(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=2)
        assert all(isinstance(r, BM25Result) for r in results)

    def test_search_rank_is_sequential(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_search_scores_are_descending(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_respected(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=2)
        assert len(results) == 2

    def test_search_empty_query_returns_empty(self, retriever: BM25Retriever) -> None:
        results = retriever.search("", top_k=5)
        assert results == []

    def test_search_whitespace_query_returns_empty(self, retriever: BM25Retriever) -> None:
        results = retriever.search("   ", top_k=5)
        assert results == []

    def test_search_no_build_raises(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        with pytest.raises(RuntimeError, match="not built"):
            r.search("machine learning")

    def test_search_relevant_chunk_returns_high_score(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=5)
        top_ids = [r.chunk_id for r in results]
        assert "c1" in top_ids or "c3" in top_ids or "c5" in top_ids

    def test_search_result_contains_all_fields(self, retriever: BM25Retriever) -> None:
        results = retriever.search("machine learning", top_k=1)
        r = results[0]
        assert r.chunk_id
        assert isinstance(r.score, float)
        assert r.rank >= 1
        assert r.arxiv_id
        assert r.title
        assert isinstance(r.authors, list)
        assert r.section_label
        assert r.page_start >= 1
        assert r.page_end >= r.page_start
        assert isinstance(r.text, str)


# ── BM25Retriever.persist + load ─────────────────────────────────────────────


class TestBM25RetrieverPersistence:
    def test_persist_creates_file(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        index_path = tmp_path / "idx.pkl"
        r = BM25Retriever(sample_chunks, index_path=index_path)
        r.build()
        r.persist()
        assert index_path.exists()

    def test_persist_returns_stats(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        index_path = tmp_path / "idx.pkl"
        r = BM25Retriever(sample_chunks, index_path=index_path)
        r.build()
        stats = r.persist()
        assert "file_path" in stats
        assert "file_size_bytes" in stats
        assert "file_size_kb" in stats
        assert stats["file_size_bytes"] > 0

    def test_persist_without_build_raises(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks, index_path=tmp_path / "idx.pkl")
        with pytest.raises(RuntimeError, match="not built"):
            r.persist()

    def test_load_returns_retriever(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        index_path = tmp_path / "idx.pkl"
        r = BM25Retriever(sample_chunks, index_path=index_path)
        r.build()
        r.persist()
        loaded = BM25Retriever.load(index_path)
        assert isinstance(loaded, BM25Retriever)
        assert len(loaded.chunks) == 5

    def test_load_search_returns_same_results(
        self, tmp_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        index_path = tmp_path / "idx.pkl"
        r = BM25Retriever(sample_chunks, index_path=index_path)
        r.build()
        r.persist()
        loaded = BM25Retriever.load(index_path)
        original = r.search("machine learning", top_k=3)
        restored = loaded.search("machine learning", top_k=3)
        assert [r.chunk_id for r in original] == [r.chunk_id for r in restored]

    def test_load_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            BM25Retriever.load(tmp_path / "nonexistent.pkl")

    def test_load_preserves_k1_b(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        index_path = tmp_path / "idx.pkl"
        r = BM25Retriever(sample_chunks, k1=2.0, b=0.5, index_path=index_path)
        r.build()
        r.persist()
        loaded = BM25Retriever.load(index_path)
        assert loaded.k1 == 2.0
        assert loaded.b == 0.5


# ── BM25Retriever tokenization ───────────────────────────────────────────────


class TestBM25RetrieverTokenization:
    def test_tokenize_lowercases(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        tokens = r._tokenize("MACHINE Learning")
        assert tokens == ["machine", "learning"]

    def test_tokenize_splits_on_whitespace(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        tokens = r._tokenize("hello world  test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_empty_string(self, sample_chunks: list[Chunk]) -> None:
        r = BM25Retriever(sample_chunks)
        assert r._tokenize("") == []
