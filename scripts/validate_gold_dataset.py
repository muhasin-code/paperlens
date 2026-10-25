#!/usr/bin/env python3
"""Validate gold dataset schema, and chunk ids against Chroma when requested."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.evaluation.gold import load_gold_dataset, validate_dataset
from src.paperlens.settings import get_settings


def _chunk_ids(rows: list[dict]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for item in rows:
        for key in ("relevant_chunk_ids", "source_chunk_ids"):
            for chunk_id in item.get(key) or []:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    found.append(chunk_id)
    return found


def _check_chroma(rows: list[dict]) -> list[str]:
    from src.paperlens.embedding.vector_store import VectorStore

    settings = get_settings()
    store = VectorStore(settings)
    missing: list[str] = []
    chunk_ids = _chunk_ids(rows)
    print(f"Checking {len(chunk_ids)} unique chunk IDs...")
    batch_size = 100
    for start in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[start : start + batch_size]
        try:
            result = store.collection.get(ids=batch, include=[])
            found_ids = set(result["ids"])
        except Exception as exc:
            print(f"Error checking batch: {exc}")
            missing.extend(batch)
            continue
        for chunk_id in batch:
            if chunk_id not in found_ids:
                missing.append(chunk_id)

    arxiv_ids: set[str] = set()
    for item in rows:
        for arxiv_id in item.get("relevant_arxiv_ids") or []:
            arxiv_ids.add(arxiv_id)
    if arxiv_ids:
        print(f"\nChecking {len(arxiv_ids)} arXiv IDs...")
        for arxiv_id in sorted(arxiv_ids):
            result = store.collection.get(where={"arxiv_id": arxiv_id}, limit=1, include=[])
            if not result["ids"]:
                print(f"   No chunks found for arXiv ID: {arxiv_id}")
            else:
                print(f"   {arxiv_id} - {len(result['ids'])} chunks found")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate eval/gold_dataset.jsonl")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--min-generation", type=int, default=75)
    parser.add_argument("--gold", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    gold_path = args.gold or Path(settings.gold_dataset_path)
    print(f"Loading gold dataset from {gold_path}...")
    rows = load_gold_dataset(gold_path)
    print(f"Loaded {len(rows)} queries")

    errors = validate_dataset(rows, min_generation=args.min_generation)
    if errors:
        print("SCHEMA VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Schema validation passed")
    if args.schema_only:
        return 0

    print("Connecting to ChromaDB...")
    missing = _check_chroma(rows)
    if missing:
        print(f"\n VALIDATION FAILED: {len(missing)} chunk IDs NOT FOUND in ChromaDB:")
        for chunk_id in missing:
            print(f"  - {chunk_id}")
        return 1
    print("\n VALIDATION PASSED: chunk IDs exist in ChromaDB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
