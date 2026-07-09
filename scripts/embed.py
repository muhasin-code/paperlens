#!/usr/bin/env python3
"""
CLI entry point for the PaperLens embedding pipeline.

Examples:
    python scripts/embed.py              # full embed using .env defaults
    python scripts/embed.py --dry-run    # embed 5 chunks in memory, no writes
    python scripts/embed.py --limit 100  # embed only the first 100 chunks (smoke test)
    python scripts/embed.py --force      # re-embed everything (upsert by id)
    python scripts/embed.py --rebuild    # reset collection, then embed all
    make embed                           # same as first option
    make embed-dry                       # same as --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.pipeline import EmbeddingPipeline
from src.paperlens.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks and store them in ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + embed in memory but do not write to ChromaDB.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N chunks (smoke test).")
    parser.add_argument(
        "--force", action="store_true", help="Re-embed all chunks (idempotent upsert by id)."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Reset the ChromaDB collection before embedding."
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = EmbeddingPipeline(settings)
    stats = pipeline.run(
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
        rebuild=args.rebuild,
    )

    print("\n--- Embedding Stats ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
