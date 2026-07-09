"""
Embedding pipeline: orchestrates chunk load -> embed -> persist in ChromaDB.

Design decisions:
- Idempotent: existing chunk_ids in the store are skipped unless --force/--rebuild.
- --rebuild fully resets the collection before embedding (clean reproducible index).
- Resumable: a checkpoint file tracks successfully persisted chunk_ids.
- Chunks streamed from Parquet in bulk; embedded in batches with tqdm progress.
"""

import logging
from pathlib import Path

import pandas as pd

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")


class EmbeddingPipeline:
    """Orchestrates the full embed-and-store pipeline over the chunk corpus."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = EmbeddingModel(settings)
        self.store = VectorStore(settings)
        self.checkpoint_path: Path = settings.embedding_checkpoint_path
        self.logger = logger

    def _load_chunks(self, limit: int | None = None) -> list[Chunk]:
        path = self.settings.processed_chunks_path
        if not path.exists():
            self.logger.error("Chunks file not found: %s", path)
            return []
        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit)
        return [Chunk.from_parquet_row(row.to_dict()) for _, row in df.iterrows()]

    def _load_checkpoint(self) -> set[str]:
        """Load chunk_ids that have already been persisted."""
        if not self.checkpoint_path.exists():
            return set()
        ids: set[str] = set()
        with open(self.checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
        self.logger.info("Loaded %d completed chunk_ids from checkpoint", len(ids))
        return ids

    def _append_checkpoint(self, chunk_ids: list[str]) -> None:
        """Append newly completed chunk_ids to the checkpoint file."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            for cid in chunk_ids:
                f.write(f"{cid}\n")

    def _clear_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def run(
        self,
        dry_run: bool = False,
        limit: int | None = None,
        force: bool = False,
        rebuild: bool = False,
    ) -> dict:
        """Execute the embedding pipeline.

        Args:
            dry_run: Load + embed a sample in memory but do NOT write to ChromaDB.
            limit: Only process the first N chunks (smoke test).
            force: Re-embed all chunks (ignore checkpoint, still upsert by id).
            rebuild: Reset the collection and checkpoint before embedding.

        Returns:
            Dict of stats: mode, chunks_loaded, chunks_embedded, vectors_in_store, output_dir.
        """
        self.logger.info(
            "=== PaperLens Embedding Start | dry_run=%s | limit=%s | force=%s | rebuild=%s ===",
            dry_run,
            limit,
            force,
            rebuild,
        )

        if rebuild and not dry_run:
            self.store.reset()
            self._clear_checkpoint()

        chunks = self._load_chunks(limit=limit)
        if not chunks:
            self.logger.warning("No chunks loaded - aborting.")
            return {
                "mode": "dry_run" if dry_run else "full",
                "chunks_loaded": 0,
                "chunks_embedded": 0,
                "vectors_in_store": self.store.count(),
                "output_dir": str(self.settings.chroma_persist_dir),
            }

        completed_ids = set() if force else self._load_checkpoint()
        remaining = [c for c in chunks if c.chunk_id not in completed_ids]

        if not remaining:
            self.logger.info("All chunks already embedded (checkpoint hit).")
            return {
                "mode": "dry_run" if dry_run else "full",
                "chunks_loaded": len(chunks),
                "chunks_embedded": 0,
                "vectors_in_store": self.store.count(),
                "output_dir": str(self.settings.chroma_persist_dir),
            }

        self.logger.info(
            "Loaded %d chunks, %d already embedded, %d remaining.",
            len(chunks),
            len(completed_ids),
            len(remaining),
        )

        UPSERT_BATCH = 5000
        total_embedded = 0

        for i in range(0, len(remaining), UPSERT_BATCH):
            batch = remaining[i : i + UPSERT_BATCH]
            embedded = self.embedder.embed_chunks(batch)

            if not dry_run:
                self.store.upsert(embedded)
                self._append_checkpoint([ec.chunk.chunk_id for ec in embedded])

            total_embedded += len(embedded)
            self.logger.info(
                "Progress: %d/%d chunks embedded",
                total_embedded,
                len(remaining),
            )

        self.logger.info(
            "=== Embedding Complete | loaded=%d | embedded=%d | store_total=%d ===",
            len(chunks),
            total_embedded,
            self.store.count(),
        )

        return {
            "mode": "dry_run" if dry_run else "full",
            "chunks_loaded": len(chunks),
            "chunks_embedded": total_embedded,
            "vectors_in_store": self.store.count(),
            "output_dir": str(self.settings.chroma_persist_dir),
        }
