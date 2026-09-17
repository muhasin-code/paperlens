"""Extraction pipeline for generating SFT dataset."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.paperlens.extraction.annotator import ChunkAnnotator
from src.paperlens.extraction.models import ExtractionExample
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.extraction")


class ExtractionPipeline:
    """Pipeline for generating extraction training dataset."""

    SECTION_LABELS = {"method", "experiments", "results", "evaluation"}

    def __init__(self, settings: Settings, annotator: ChunkAnnotator):
        self.settings = settings
        self.annotator = annotator

    def _load_checkpoint(self) -> set[str]:
        """Load set of already-annotated chunk IDs."""
        checkpoint_path = self.settings.extraction_checkpoint_path
        if not checkpoint_path.exists():
            return set()

        annotated = set()
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        annotated.add(data["chunk_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        logger.info("Loaded %d annotated chunk IDs from checkpoint", len(annotated))
        return annotated

    def _save_checkpoint_entry(self, chunk_id: str) -> None:
        """Append a single chunk_id to the checkpoint file."""
        checkpoint_path = self.settings.extraction_checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk_id": chunk_id}) + "\n")

    def _load_chunks(self) -> list[Chunk]:
        """Load chunks from the parquet file."""
        chunks_path = self.settings.processed_chunks_path
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        df = pd.read_parquet(chunks_path)
        chunks = [Chunk(**row) for row in df.to_dict("records")]
        logger.info("Loaded %d total chunks from %s", len(chunks), chunks_path)
        return chunks

    def _filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter to extractable sections only."""
        filtered = [c for c in chunks if c.section_label in self.SECTION_LABELS]
        logger.info(
            "Filtered to %d chunks in extractable sections (method, experiments, results, evaluation)",
            len(filtered),
        )
        return filtered

    def _sample_chunks(self, chunks: list[Chunk], max_examples: int) -> list[Chunk]:
        """Sample up to max_examples chunks."""
        if len(chunks) <= max_examples:
            return chunks

        sampled = random.sample(chunks, max_examples)
        logger.info("Sampled %d chunks from %d candidates", len(sampled), len(chunks))
        return sampled

    def _split_train_val(
        self, examples: list[ExtractionExample]
    ) -> tuple[list[ExtractionExample], list[ExtractionExample]]:
        """Split examples into train and validation sets."""
        val_split = self.settings.extraction_val_split
        val_size = int(len(examples) * val_split)

        random.shuffle(examples)
        val = examples[:val_size]
        train = examples[val_size:]

        logger.info("Split %d examples into train=%d, val=%d", len(examples), len(train), len(val))
        return train, val

    def _write_jsonl(self, examples: list[ExtractionExample], path: Path) -> None:
        """Write examples to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(ex.to_jsonl())

        logger.info("Wrote %d examples to %s", len(examples), path)

    async def run(
        self,
        limit: int | None = None,
        force: bool = False,
        sections: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run the extraction pipeline.

        Args:
            limit: Maximum number of examples to process (default: settings.extraction_max_examples)
            force: If True, ignore checkpoint and re-annotate all chunks
            sections: Override which sections to include (default: method,experiments,results,evaluation)
            dry_run: If True, only annotate 5 examples and don't write files

        Returns:
            Summary dict with counts and output paths
        """
        max_examples = limit or self.settings.extraction_max_examples
        target_sections = (
            set(sections) if sections else set(self.settings.extraction_sections.split(","))
        )

        if sections:
            self.SECTION_LABELS = set(sections)

        logger.info(
            "Starting extraction pipeline: max_examples=%d, sections=%s",
            max_examples,
            target_sections,
        )

        all_chunks = self._load_chunks()
        filtered_chunks = [c for c in all_chunks if c.section_label in target_sections]
        sampled_chunks = self._sample_chunks(filtered_chunks, max_examples)

        if force:
            annotated_ids = set()
            checkpoint_path = self.settings.extraction_checkpoint_path
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Force mode: cleared checkpoint")
        else:
            annotated_ids = self._load_checkpoint()

        to_annotate = [c for c in sampled_chunks if c.chunk_id not in annotated_ids]
        logger.info(
            "Annotating %d new chunks (skipped %d already annotated)",
            len(to_annotate),
            len(annotated_ids),
        )

        if dry_run:
            to_annotate = to_annotate[:5]
            logger.info("Dry run: limiting to 5 examples")

        examples: list[ExtractionExample] = []
        failed: list[str] = []

        for chunk in tqdm(to_annotate, desc="Annotating chunks"):
            try:
                example = await self.annotator.annotate(chunk)
                examples.append(example)

                self._save_checkpoint_entry(chunk.chunk_id)
            except Exception as e:
                logger.error("Failed to annotate %s: %s", chunk.chunk_id, e)
                failed.append(chunk.chunk_id)

        train_examples, val_examples = self._split_train_val(examples)

        summary = {
            "total_chunks_loaded": len(all_chunks),
            "chunks_filtered": len(filtered_chunks),
            "chunks_sampled": len(sampled_chunks),
            "already_annotated": len(annotated_ids),
            "new_chunks_to_annotate": len(to_annotate),
            "successfully_annotated": len(examples),
            "failed": len(failed),
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "failed_chunk_ids": failed,
        }

        if not dry_run:
            train_path = self.settings.extraction_train_path
            val_path = self.settings.extraction_val_path

            if train_examples:
                self._write_jsonl(train_examples, train_path)
            if val_examples:
                self._write_jsonl(val_examples, val_path)

            logger.info("Dataset written to %s and %s", train_path, val_path)

        return summary
