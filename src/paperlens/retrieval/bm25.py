"""BM25 keyword retrieval service for PaperLens.

Provides BM25-based keyword search over the chunk corpus using rank_bm25.
Persists index to data/bm25_index.pkl for fast loading.
"""

import time
from pathlib import Path

import joblib
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from src.paperlens.parsing.models import Chunk


class BM25Result(BaseModel):
    """Result from BM25 keyword search."""

    chunk_id: str
    score: float
    rank: int
    arxiv_id: str
    title: str
    authors: list[str]
    section_label: str
    page_start: int
    page_end: int
    text: str


class BM25Retriever:
    """BM25 keyword retriever over chunk corpus.

    Uses rank_bm25.BM25Okapi with default parameters (k1=1.5, b=0.75).
    Tokenization is simple whitespace + lowercase for speed on CPU-only systems.
    """

    INDEX_PATH = Path("data/bm25_index.pkl")
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(self, chunks: list[Chunk], k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        """Initialize BM25 retriever with chunks.

        Args:
            chunks: List of Chunk objects to index
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        self._build_time_ms: float = 0.0
        self._build_start: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        return text.lower().split()

    def build(self) -> dict:
        """Build BM25 index from chunks.

        Returns:
            Dict with build statistics (corpus_size, vocab_size, build_time_ms)
        """
        self._build_start = time.perf_counter()

        self._tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        self._build_time_ms = (time.perf_counter() - self._build_start) * 1000

        vocab: set[str] = set()
        for tokens in self._tokenized_corpus:
            vocab.update(tokens)

        return {
            "corpus_size": len(self.chunks),
            "vocab_size": len(vocab),
            "build_time_ms": self._build_time_ms,
            "k1": self.k1,
            "b": self.b,
        }

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        """Search for top-k chunks matching the query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of BM25Result objects sorted by score (descending)
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        if not query.strip():
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                BM25Result(
                    chunk_id=chunk.chunk_id,
                    score=float(scores[idx]),
                    rank=rank,
                    arxiv_id=chunk.arxiv_id,
                    title=chunk.title,
                    authors=chunk.authors,
                    section_label=chunk.section_label,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                )
            )

        return results

    def persist(self) -> dict:
        """Persist BM25 index to disk.

        Returns:
            Dict with persistence statistics (file_size_bytes, file_size_kb)
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        persist_data = {
            "chunks": self.chunks,
            "bm25": self._bm25,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
            "build_time_ms": self._build_time_ms,
        }

        self.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(persist_data, self.INDEX_PATH)

        file_size = self.INDEX_PATH.stat().st_size

        return {
            "file_path": str(self.INDEX_PATH),
            "file_size_bytes": file_size,
            "file_size_kb": file_size / 1024,
        }

    @classmethod
    def load(cls, index_path: Path | None = None) -> "BM25Retriever":
        """Load BM25 index from disk.

        Args:
            index_path: Path to index file (default: data/bm25_index.pkl)

        Returns:
            BM25Retriever instance with loaded index
        """
        path = Path(index_path) if index_path else cls.INDEX_PATH

        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")

        persist_data = joblib.load(path)

        retriever = cls.__new__(cls)
        retriever.chunks = persist_data["chunks"]
        retriever._bm25 = persist_data["bm25"]
        retriever._tokenized_corpus = persist_data["tokenized_corpus"]
        retriever.k1 = persist_data["k1"]
        retriever.b = persist_data["b"]
        retriever._build_time_ms = persist_data["build_time_ms"]
        retriever._build_start = 0

        return retriever
