#!/usr/bin/env python3
"""Smoke test for hybrid retrieval with RRF."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.settings import get_settings


def main() -> int:
    settings = get_settings()
    r = HybridRetriever(settings)
    results = r.search("machine learning", top_k=5)
    print(f"Hybrid retrieval returned {len(results)} results")
    print(f"Top-1: {results[0].chunk.chunk_id} (RRF score={results[0].score:.6f})")
    print("Hybrid retrieval OK")
    return 0


if __name__ == "__main__":
    exit(main())
