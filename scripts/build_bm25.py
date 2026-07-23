#!/usr/bin/env python3
"""Build BM25 keyword index from parsed chunks.

Usage:
    python scripts/build_bm25.py [--rebuild]
    make bm25-build

This script loads chunks from data/processed/chunks.parquet and builds a BM25 index
saved to data/bm25_index.pkl.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.paperlens.parsing.models import Chunk
from src.paperlens.retrieval.bm25 import BM25Retriever
from src.paperlens.settings import get_settings


def main() -> int:
    settings = get_settings()
    chunks_path = settings.processed_chunks_path
    index_path = settings.bm25_index_path

    parser = argparse.ArgumentParser(description="Build BM25 keyword index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}", file=sys.stderr)
        print("Run 'make parse' first to generate chunks.", file=sys.stderr)
        return 1

    print(f"Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    chunks = [Chunk(**row) for row in df.to_dict("records")]
    print(f"Loaded {len(chunks)} chunks")

    if index_path.exists() and not args.rebuild:
        print(f"Index already exists at {index_path}. Use --rebuild to regenerate.")
        return 0

    print("Building BM25 index...")
    start = time.perf_counter()

    retriever = BM25Retriever(chunks)
    build_stats = retriever.build()
    persist_stats = retriever.persist()

    elapsed = time.perf_counter() - start

    print("\n=== BM25 Index Built ===")
    print(f"Corpus size: {build_stats['corpus_size']} chunks")
    print(f"Vocabulary size: {build_stats['vocab_size']} unique terms")
    print(f"Build time: {build_stats['build_time_ms']:.1f} ms")
    print(f"Index file: {persist_stats['file_path']}")
    print(
        f"Index size: {persist_stats['file_size_kb']:.1f} KB ({persist_stats['file_size_bytes']} bytes)"
    )
    print(f"Total time: {elapsed:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
