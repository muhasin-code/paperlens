#!/usr/bin/env python3
"""Validate gold dataset: all chunk IDs must exist in ChromaDB."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.settings import get_settings


def load_gold_dataset(path: Path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    settings = get_settings()
    gold_path = Path(settings.gold_dataset_path)

    print(f"Loading gold dataset from {gold_path}...")
    gold_dataset = load_gold_dataset(gold_path)
    print(f"Loaded {len(gold_dataset)} queries")

    print("Connecting to ChromaDB...")
    store = VectorStore(settings)

    all_chunk_ids = set()
    missing = []

    for item in gold_dataset:
        for chunk_id in item.get("relevant_chunk_ids", []):
            all_chunk_ids.add(chunk_id)

    print(f"Checking {len(all_chunk_ids)} unique chunk IDs...")

    # Batch check
    batch_size = 100
    chunk_id_list = list(all_chunk_ids)

    for i in range(0, len(chunk_id_list), batch_size):
        batch = chunk_id_list[i : i + batch_size]
        try:
            result = store.collection.get(ids=batch, include=[])
            found_ids = set(result["ids"])
            for cid in batch:
                if cid not in found_ids:
                    missing.append(cid)
        except Exception as e:
            print(f"Error checking batch: {e}")
            missing.extend(batch)

    if missing:
        print(f"\n❌ VALIDATION FAILED: {len(missing)} chunk IDs NOT FOUND in ChromaDB:")
        for cid in missing:
            print(f"  - {cid}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED: All {len(all_chunk_ids)} chunk IDs exist in ChromaDB")

        # Also check paper-level IDs if present
        arxiv_ids = set()
        for item in gold_dataset:
            for arxiv_id in item.get("relevant_arxiv_ids", []):
                arxiv_ids.add(arxiv_id)

        if arxiv_ids:
            print(f"\nChecking {len(arxiv_ids)} arXiv IDs...")
            # Query for papers
            for arxiv_id in arxiv_ids:
                result = store.collection.get(where={"arxiv_id": arxiv_id}, limit=1, include=[])
                if not result["ids"]:
                    print(f"  ⚠️  No chunks found for arXiv ID: {arxiv_id}")
                else:
                    print(f"  ✅ {arxiv_id} - {len(result['ids'])} chunks found")


if __name__ == "__main__":
    main()
