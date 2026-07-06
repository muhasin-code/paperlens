"""
Parsing Pipeline: orchestrates PDF parsing, chunking, and Parquet persistance.

Design decisions:
- Idempotent: existing chunk output is loaded; already-processed papers are skipped unless --force is passed.
- Parquet for persistance: columnar, compressed, fast for batch reads in Phase 1.3.
- Per-paper progress logging so long runs are monitorable.
"""

import logging

import pandas as pd

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.chunker import SectionAwareChunker
from src.paperlens.parsing.models import Chunk
from src.paperlens.parsing.pdf_parser import PdfParser
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.parsing")


class ParsingPipeline:
    """Orchestrates the full parse-and-chunk pipeline over the paper Corpus."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = PdfParser()
        self.chunker = SectionAwareChunker(settings=settings)
        self.logger = logger

    def _load_existing_chunk_ids(self) -> set[str]:
        """Return the set of chunk_ids already in the output Parquet file."""
        path = self.settings.processed_chunks_path
        if not path.exists():
            return set()
        try:
            df = pd.read_parquet(path, columns=["chunk_id"])
            return set(df["chunk_id"].tolist())
        except Exception as exc:
            self.logger.warning("Could not read existing chunks: %s", exc)
            return set()

    def _load_papers_with_pdfs(self) -> list[Paper]:
        """Load papers from metadata.jsonl that have a downloaded PDF."""
        if not self.settings.metadata_path.exists():
            self.logger.error("Metadata file not found: %s", self.settings.metadata_path)
            return []

        papers: list[Paper] = []
        with self.settings.metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = Paper.from_jsonl(line)
                    if paper.pdf_path is not None:
                        papers.append(paper)
                except Exception as exc:
                    self.logger.warning("Could not parse metadata line: %s", exc)

        self.logger.info("Loaded %d papers with PDFs.", len(papers))
        return papers

    def _append_chunks_to_parquet(self, chunks: list[Chunk]) -> None:
        """Append chunks to the Parquet file (creates if absent)."""
        if not chunks:
            return

        path = self.settings.processed_chunks_path
        path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([c.to_parquet_row() for c in chunks])

        if path.exists():
            existing_df = pd.read_parquet(path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)

    def run(self, dry_run: bool = False, limit: int | None = None, force: bool = False) -> dict:
        """
        Executes the full parsing pipeline.

        Args:
            dry_run: If True, parse and chunk but do not write to disk.
            limit: If set, only process the first N papers (for smoke tests).
            force: If True, re-process papers even if their chunks already exists

        Returns:
            Dict of stats: mode, papers_processed, papers_skipped, total_chunks, avg_chunks_per_paper, output_path.
        """
        self.logger.info(
            "=== PaperLens Parsing Start | dry_run=%s | limit=%s | force=%s ===",
            dry_run,
            limit,
            force,
        )

        papers = self._load_papers_with_pdfs()
        if limit is not None:
            papers = papers[:limit]

        existing_ids = set() if force else self._load_existing_chunk_ids()

        papers_processed = 0
        papers_skipped = 0
        all_new_chunks: list[Chunk] = []

        for paper in papers:
            # Check if any chunks from this paper already exists.
            paper_prefix = f"{paper.arxiv_id}_chunk_"
            has_existing = any(cid.startswith(paper_prefix) for cid in existing_ids)

            if has_existing and not force:
                self.logger.debug("Skip (laready parsed): %s", paper.arxiv_id)
                papers_skipped += 1
                continue

            sections = self.parser.parse_paper(paper)
            if not sections:
                self.logger.warning("No sections detected for %s - skipping.", paper.arxiv_id)
                papers_skipped += 1
                continue

            chunks = self.chunker.chunk_paper(paper, sections)
            all_new_chunks.extend(chunks)
            papers_processed += 1
            self.logger.debug(
                "Parse %s: %d sections -> %d chunks", paper.arxiv_id, len(sections), len(chunks)
            )

        if not dry_run and all_new_chunks:
            self._append_chunks_to_parquet(all_new_chunks)

        avg_chunks = len(all_new_chunks) / papers_processed if papers_processed > 0 else 0.0

        self.logger.info(
            "=== Parsing Complete | processed=%d | skipped=%d | new_chunks=%d | abg=%.1f ===",
            papers_processed,
            papers_skipped,
            len(all_new_chunks),
            avg_chunks,
        )

        return {
            "mode": "dry_run" if dry_run else "full",
            "papers_processed": papers_processed,
            "papers_skipped": papers_skipped,
            "total_chunks": len(all_new_chunks),
            "avg_chunks_per_paper": round(avg_chunks, 1),
            "output_path": str(self.settings.processed_chunks_path),
        }
