"""
Semantic retriever: query embedding + top-k search -> ranked Chunk results.

Design decisions:
- Reuses EmbeddingModel (query prefix) and VectorStore (cosine search).
- Reconstructs Chunk Pydantic objects from ChromaDB metadata so downstream
  code receives the same typed contract as parsing produced.
- ChromaDB returns cosine DISTANCE; we convert to similarity score = 1 - distance.
"""

import logging

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


class SemanticRetriever:
    """Top-k semantic search over the ChromaDB chunk index."""

    def __init__(
        self,
        settings: Settings,
        top_k: int | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or EmbeddingModel(settings)
        self.store = VectorStore(settings)
        self.top_k = top_k or settings.retrieval_top_k
        self.logger = logging.getLogger("paperlens.embedding")

    @staticmethod
    def _chunk_from_metadata(chunk_id: str, metadata: dict, document: str) -> Chunk:
        authors = [a.strip() for a in metadata.get("authors", "").split(";") if a.strip()]
        return Chunk(
            chunk_id=chunk_id,
            arxiv_id=metadata["arxiv_id"],
            title=metadata.get("title", ""),
            authors=authors,
            section_label=metadata.get("section_label", "unknown"),
            chunk_index=int(metadata.get("chunk_index", 0)),
            page_start=int(metadata.get("page_start", 0)),
            page_end=int(metadata.get("page_end", 0)),
            text=document,
            token_count=int(metadata.get("token_count", 0)),
            total_chunks_in_section=int(metadata.get("total_chunks_in_section", 0)),
            total_chunks_in_paper=int(metadata.get("total_chunks_in_paper", 0)),
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return ranked RetrievalResult objects for a natural-language query."""
        k = top_k or self.top_k
        qvec = self.embedder.embed_query(query)
        resp = self.store.query(qvec, top_k=k)

        ids = resp["ids"][0]
        docs = resp["documents"][0]
        metas = resp["metadatas"][0]
        dists = resp["distances"][0]

        results: list[RetrievalResult] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists, strict=False), start=1
        ):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            chunk = self._chunk_from_metadata(cid, meta, doc)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results
