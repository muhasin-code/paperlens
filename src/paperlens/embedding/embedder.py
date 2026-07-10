"""
Embedding model wrapper around sentence-transformers.

Design decisions:
- BAAI/bge-large-en-v1.5: 1024-dim, strong general purpose English retriever.
- Query prefix applied ONLY to queries (BGE training convention).
- Embedding normalized to unit length so dot-product == cosine similarity.
- CPU-only by default; batch encoding with tqdm progress.
"""

import logging

from sentence_transformers import SentenceTransformer

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")

# BGE training convention: prefix queries with this string.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"

# Default batch size for corpus encoding (memory-safe on CPU).
DEFAULT_BATCH_SIZE = 16


class EmbeddingModel:
    """Wraps a sentence-transformers model for chunk + query embedding."""

    def __init__(self, settings: Settings, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_name = settings.embedding_model
        self.batch_size = batch_size
        self.device = "cpu"  # CPU-only box; override via settings if GPU availale
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self._model.get_embedding_dimension()
        self.logger = logger
        self.logger.info(
            "Loaded embedding model %s (dim=%d, device=%s)",
            self.model_name,
            self.dimension,
            self.device,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of corpus chunks (no query prefix, normalized)."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [
            EmbeddedChunk(chunk=c, embedding=vec.tolist(), dimension=self.dimension)
            for c, vec in zip(chunks, vectors, strict=False)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string WITH the BGE query prefix, normalized."""
        prefixed = f"{BGE_QUERY_PREFIX}{query}"
        vec = self._model.encode(
            [prefixed],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries (prefixed, normalized)."""
        prefixed = [f"{BGE_QUERY_PREFIX}{q}" for q in queries]
        vecs = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]
