#!/usr/bin/env python3
"""
CLI entrypoint for the PaperLens arXiv ingestion pipeline.

Examples:
    python scripts/ingest.py                    # full run using .env defaults
    python scripts/ingest.py --dry-run          # preview without downloading
    python scripts/ingest.py --max-results 50   # quick smoke test
    make ingest                                 # same as first option
    make ingest-dry                             # same as --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.ingestion.fetcher import ArxivFetcher
from src.paperlens.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into the PaperLens dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        type=str,
        help="arXiv category (overrides ARXIV_CATEGORY in .env).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Max papers to fetch (overrides ARXIV_MAX_RESULTS in .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata only. No PDFs downloaded; nothing written to disk.",
    )
    args = parser.parse_args()

    settings = get_settings()

    overrides: dict = {}
    if args.category:
        overrides["arxiv_category"] = args.category
    if args.max_results:
        overrides["arxiv_max_results"] = args.max_results
    if overrides:
        settings = settings.nodel_copy(update=overrides)

    fetcher = ArxivFetcher(settings=settings)
    stats = fetcher.run(dry_run=args.dry_run)

    print("\n--- Ingestion Stats ---")
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
