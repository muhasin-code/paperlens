#!/usr/bin/env python3
"""
CLI entry point for the PaperLens parsing pipeline.

Examples:
    python scripts/parse.py             # full parsing using .env defaults.
    python scripts/parse.py --dry-run   # preview without writing
    python scripts/parse.py --limit 5   # parse only 5 papers (smoke test)
    python scripts/parse.py --force     # re-parse all papers
    make parse                          # same as first option
    make parse-dry                      # same as --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.parsing.pipeline import ParsingPipeline
from src.paperlens.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PDFs into section-aware chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and chunk but do not write to disk."
    )
    parser.add_argument(
        "--limit", type=int, help="Only process the first N papers (for smoke test)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-parse papers wven if their chunks already exists."
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = ParsingPipeline(settings=settings)
    stats = pipeline.run(dry_run=args.dry_run, limit=args.limit, force=args.force)

    print("\n--- Parsing Stats ---")
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
