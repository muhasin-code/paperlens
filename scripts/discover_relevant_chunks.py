#!/usr/bin/env python3
"""Discover relevant chunks for gold dataset queries.

Runs BOTH semantic and BM25 search for manual relevance labeling.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.retrieval.bm25 import BM25Retriever
from src.paperlens.settings import get_settings


def load_gold_dataset(path: Path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def search_semantic(retriever, query: str, top_k: int = 10):
    results = retriever.search(query=query, top_k=top_k)
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "score": r.score,
            "text": r.chunk.text[:200],
            "arxiv_id": r.chunk.arxiv_id,
            "title": r.chunk.title[:50] if r.chunk.title else "N/A",
            "section_label": r.chunk.section_label,
            "source": "semantic",
        }
        for r in results
    ]


def search_bm25(retriever, query: str, top_k: int = 10):
    results = retriever.search(query=query, top_k=top_k)
    return [
        {
            "chunk_id": r.chunk_id,
            "score": r.score,
            "text": r.text[:200],
            "arxiv_id": r.arxiv_id,
            "title": r.title[:50] if r.title else "N/A",
            "section_label": r.section_label,
            "source": "bm25",
        }
        for r in results
    ]


def main():
    settings = get_settings()
    gold_path = Path(settings.gold_dataset_path)

    print(f"Loading gold dataset from {gold_path}...")
    gold_dataset = load_gold_dataset(gold_path)
    print(f"Loaded {len(gold_dataset)} queries\n")

    print("Initializing retrievers...")
    embedder = EmbeddingModel(settings)
    semantic_retriever = SemanticRetriever(settings, embedder=embedder)
    bm25_retriever = BM25Retriever.load(settings.bm25_index_path)
    print("Ready\n")

    all_results = {}

    for i, item in enumerate(gold_dataset, 1):
        query = item["query"]
        current_relevant = item.get("relevant_chunk_ids", [])

        print(f"{'='*80}")
        print(f"QUERY {i}/{len(gold_dataset)}: {query}")
        print(f"Current relevant_chunk_ids: {current_relevant}")
        print(f"{'='*80}")

        sem_results = search_semantic(semantic_retriever, query, top_k=50)
        bm25_results = search_bm25(bm25_retriever, query, top_k=50)
        all_results[query] = {"semantic": sem_results, "bm25": bm25_results}

        # Print semantic results
        print("\n--- SEMANTIC TOP 10 ---")
        print(f"{'Rank':<5} {'Chunk ID':<35} {'Score':<8} {'arXiv ID':<18} {'Section':<20} Text")
        print(f"{'-'*5} {'-'*35} {'-'*8} {'-'*18} {'-'*20} {'-'*50}")
        for rank, r in enumerate(sem_results, 1):
            arxiv = r["arxiv_id"][:16] if r["arxiv_id"] else "N/A"
            section = r["section_label"][:18] if r["section_label"] else "N/A"
            text_preview = r["text"][:80].replace("\n", " ")
            print(
                f"{rank:<5} {r['chunk_id']:<35} {r['score']:<8.4f} {arxiv:<18} {section:<20} {text_preview}..."
            )

        # Print BM25 results
        print("\n--- BM25 TOP 10 ---")
        print(f"{'Rank':<5} {'Chunk ID':<35} {'Score':<8} {'arXiv ID':<18} {'Section':<20} Text")
        print(f"{'-'*5} {'-'*35} {'-'*8} {'-'*18} {'-'*20} {'-'*50}")
        for rank, r in enumerate(bm25_results, 1):
            arxiv = r["arxiv_id"][:16] if r["arxiv_id"] else "N/A"
            section = r["section_label"][:18] if r["section_label"] else "N/A"
            text_preview = r["text"][:80].replace("\n", " ")
            print(
                f"{rank:<5} {r['chunk_id']:<35} {r['score']:<8.4f} {arxiv:<18} {section:<20} {text_preview}..."
            )

        # Find overlap
        sem_ids = {r["chunk_id"] for r in sem_results}
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        overlap = sem_ids & bm25_ids
        bm25_only = bm25_ids - sem_ids

        if overlap:
            print(f"\n--- OVERLAP (in both): {len(overlap)} chunks ---")
            for cid in overlap:
                print(f"  {cid}")

        if bm25_only:
            print(f"\n--- BM25 ONLY (semantic missed): {len(bm25_only)} chunks ---")
            for cid in bm25_only:
                # Find the BM25 result for this chunk
                for r in bm25_results:
                    if r["chunk_id"] == cid:
                        print(f"  {cid} (score={r['score']:.4f}) - {r['text'][:60]}...")
                        break

        print()

    output_path = Path("eval/chunk_discovery_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print("Review BOTH semantic and BM25 results. Label chunks that are genuinely relevant.")
    print("Pay special attention to BM25-ONLY chunks - hybrid needs these to shine.")
    print("Aim for 5-10 relevant chunks per query.")
    print("Add relevant_arxiv_ids for paper-level scoring.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
