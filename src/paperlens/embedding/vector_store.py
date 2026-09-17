"""
ChromaDB vector store for embedded chunks.

Design decisions:
- Single collection 'paperlens_chunks', persisted to settings.chroma_persist_dir.
- chunk_id is the ChromaDB id -> upsert is naturally idempotent.
- Metadata is flattened (authors joined) because ChromaDB rejects list values.
- cosine space so score increases with similarity.
"""

import logging
from pathlib import Path

import chromadb

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")

COLLECTION_NAME = "paperlens_chunks"


class VectorStore:
    """Persistent ChromaDB store for embedded chunks."""

    def __init__(self, settings: Settings) -> None:
        self.persist_dir: Path = settings.chroma_persist_dir
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None
        self.logger = logger

    def _ensure_initialized(self) -> None:
        """Lazily initialize ChromaDB client and collection on first use."""
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info("Initialized ChromaDB collection: %s", COLLECTION_NAME)

    @property
    def client(self) -> chromadb.PersistentClient:
        self._ensure_initialized()
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        self._ensure_initialized()
        return self._collection

    @staticmethod
    def _metadata_for(ec: EmbeddedChunk) -> dict:
        c = ec.chunk
        return {
            "arxiv_id": c.arxiv_id,
            "title": c.title,
            "authors": "; ".join(c.authors),
            "section_label": c.section_label,
            "chunk_index": c.chunk_index,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "total_chunks_in_section": c.total_chunks_in_section,
            "total_chunks_in_paper": c.total_chunks_in_paper,
            "dimension": ec.dimension,
        }

    def upsert(self, embedded: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks (idempotent by chunk_id)."""
        if not embedded:
            return
        batch_size = 5000  # Stay under ChromaDB's ~5461 limit
        for i in range(0, len(embedded), batch_size):
            batch = embedded[i : i + batch_size]
            self.collection.upsert(
                ids=[ec.chunk.chunk_id for ec in batch],
                documents=[ec.chunk.text for ec in batch],
                embeddings=[ec.embedding for ec in batch],
                metadatas=[self._metadata_for(ec) for ec in batch],
            )
        self.logger.info("Upserted %d chunks (total now %d)", len(embedded), self.count())

    def count(self) -> int:
        """Return the number of vectors currently in the collection."""
        return self.collection.count()

    def query(self, query_vector: list[float], top_k: int) -> dict:
        """Return the top_k nearest neighbours for a single query vector."""
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def reset(self) -> None:
        """Delete and recreate the collection (used by --rebuild)."""
        self._ensure_initialized()
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Reset collection %s", COLLECTION_NAME)

    def __repr__(self) -> str:
        return (
            f"VectorStore(persist_dir={self.persist_dir!r}, initialized={self._client is not None})"
        )
