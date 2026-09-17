#!/usr/bin/env python3
"""Build extraction dataset for QLoRA fine-tuning.

Usage:
    python scripts/build_extraction_dataset.py [--dry-run] [--limit N] [--force] [--sections sect1,sect2]
    make build-extraction-dataset

This script generates a supervised extraction dataset by annotating chunks
from data/processed/chunks.parquet using the configured LLM provider.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.extraction.annotator import ChunkAnnotator
from src.paperlens.extraction.pipeline import ExtractionPipeline
from src.paperlens.llm import get_llm_provider
from src.paperlens.settings import get_settings


async def async_main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Build extraction dataset for fine-tuning")
    parser.add_argument(
        "--dry-run", action="store_true", help="Annotate 5 examples only, no writes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to annotate (default: EXTRACTION_MAX_EXAMPLES)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-annotate all chunks, ignore checkpoint"
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Comma-separated list of sections (default: method,experiments,results,evaluation)",
    )
    args = parser.parse_args()

    sections = None
    if args.sections:
        sections = [s.strip() for s in args.sections.split(",")]

    print(f"Loading LLM provider ({settings.llm_provider})...")
    llm_provider = get_llm_provider(settings)
    print(f"LLM provider: {settings.llm_model or settings.ollama_model}")

    annotator = ChunkAnnotator(llm_provider, settings)
    pipeline = ExtractionPipeline(settings, annotator)

    print("\nRunning extraction pipeline...")
    if args.dry_run:
        print("  Mode: DRY RUN (5 examples only, no writes)")
    if args.limit:
        print(f"  Limit: {args.limit} examples")
    if args.force:
        print("  Force: re-annotate all chunks")
    if sections:
        print(f"  Sections: {sections}")

    summary = await pipeline.run(
        limit=args.limit,
        force=args.force,
        sections=sections,
        dry_run=args.dry_run,
    )

    print("\n=== Pipeline Summary ===")
    for key, value in summary.items():
        if key != "failed_chunk_ids":
            print(f"  {key}: {value}")

    if args.dry_run:
        print("\nDry run complete. Remove --dry-run to generate actual dataset.")
        return 0

    train_path = settings.extraction_train_path
    val_path = settings.extraction_val_path

    print("\nDataset written to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Checkpoint: {settings.extraction_checkpoint_path}")

    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
