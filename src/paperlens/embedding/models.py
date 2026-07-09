"""Pydantic models for embedded chunks and retrieval results.

EmbeddedChunk: a Chunk plus its dense vector. Kept separate from the base
Chunk model so that raw text/metadata I/O (parsing, persistence) is never
coupled to the embedding dimension.
"""

from pydantic import BaseModel

from src.paperlens.parsing.models import Chunk


class EmbeddedChunk(BaseModel):
    """A Chunk together with its dense embedding vector."""

    chunk: Chunk
    """The source chunk (full provenance retained for citation)."""

    embedding: list[float]
    """Dense vector produced by the embedding model."""

    dimension: int
    """Embedding dimensionality (e.g. 1024 for bge-large-en-v1.5)."""

    def vector(self) -> list[float]:
        """Return the embedding vector directly (convenience for ChromaDB)."""
        return self.embedding


class RetrievalResult(BaseModel):
    """A single ranked retrieval hit returned by the semantic retriever."""

    chunk: Chunk
    """The matched chunk with full provenance."""

    score: float
    """Similarity score (cosine, higher = more relevant)."""

    rank: int
    """1-based rank position in the returned list."""
