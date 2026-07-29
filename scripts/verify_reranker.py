#!/usr/bin/env python3
"""Smoke test for cross-encoder reranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import get_settings


def main() -> int:
    settings = get_settings()

    print("Running hybrid retrieval...")
    hybrid = HybridRetriever(settings)
    candidates = hybrid.search("machine learning", top_k=20)
    print(f"Hybrid retrieval returned {len(candidates)} candidates")

    print("Running cross-encoder reranking...")
    reranker = CrossEncoderReranker(settings)
    results = reranker.rerank("machine learning", candidates, top_k=5)
    print(f"Reranked results: {len(results)} items")

    if results:
        print(f"Top-1: {results[0].chunk.chunk_id} (score={results[0].score:.6f})")

    print("Reranker OK")
    return 0


if __name__ == "__main__":
    exit(main())
