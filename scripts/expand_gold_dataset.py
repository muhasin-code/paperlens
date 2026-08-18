#!/usr/bin/env python3
"""Expand gold dataset labels using semantic + BM25 discovery.

Keeps seed chunk IDs, adds relevant_arxiv_ids, and expands each query to 5-10
chunk labels by unioning top semantic and BM25 hits from the seed papers.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.evaluation.metrics import arxiv_ids_from_chunks, chunk_to_arxiv_id
from src.paperlens.retrieval.bm25 import BM25Retriever
from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import get_settings

MIN_LABELS = 5
MAX_LABELS = 15
DISCOVERY_TOP_K = 50


def load_gold_dataset(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def discover_chunks(
    query: str,
    semantic: SemanticRetriever,
    bm25: BM25Retriever,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    relevant_papers: set[str],
    seed_chunk_ids: set[str],
) -> list[str]:
    """Collect candidate chunk IDs from both retrievers for seed papers."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(chunk_id: str) -> None:
        if chunk_id in seen or chunk_to_arxiv_id(chunk_id) not in relevant_papers:
            return
        seen.add(chunk_id)
        candidates.append(chunk_id)

    for chunk_id in seed_chunk_ids:
        add(chunk_id)

    sem_hits = [r.chunk.chunk_id for r in semantic.search(query, top_k=DISCOVERY_TOP_K)]
    bm25_hits = [r.chunk_id for r in bm25.search(query, top_k=DISCOVERY_TOP_K)]
    hybrid_hits = [r.chunk.chunk_id for r in hybrid.search(query, top_k=DISCOVERY_TOP_K)]
    reranked = reranker.rerank(query=query, results=hybrid.search(query, top_k=20), top_k=10)
    reranker_hits = [r.chunk.chunk_id for r in reranked]

    # Add to interleaving loop:
    for s, b, h, r in itertools.zip_longest(sem_hits, bm25_hits, hybrid_hits, reranker_hits):
        for cid in (s, b, h, r):
            if cid:
                add(cid)

    return candidates[:MAX_LABELS] if len(candidates) >= MIN_LABELS else candidates


def expand_entry(
    entry: dict,
    semantic: SemanticRetriever,
    bm25: BM25Retriever,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> dict:
    seed_chunks = entry.get("relevant_chunk_ids") or []
    seed_set = set(seed_chunks)
    relevant_papers = set(arxiv_ids_from_chunks(seed_chunks))

    expanded_chunks = discover_chunks(
        query=entry["query"],
        semantic=semantic,
        bm25=bm25,
        hybrid=hybrid,
        reranker=reranker,
        relevant_papers=relevant_papers,
        seed_chunk_ids=seed_set,
    )

    # Ensure seeds are always kept even if discovery order dropped them.
    merged: list[str] = []
    seen: set[str] = set()
    for chunk_id in expanded_chunks + seed_chunks:
        if chunk_id not in seen:
            seen.add(chunk_id)
            merged.append(chunk_id)
    merged = merged[:MAX_LABELS]

    return {
        "query": entry["query"],
        "relevant_arxiv_ids": sorted(relevant_papers),
        "relevant_chunk_ids": merged,
        "expected_answer_summary": entry.get("expected_answer_summary", ""),
    }


def main() -> int:
    settings = get_settings()
    gold_path = Path(settings.gold_dataset_path)
    output_path = gold_path

    print(f"Loading seed gold dataset from {gold_path}...")
    seed_dataset = load_gold_dataset(gold_path)
    print(f"Loaded {len(seed_dataset)} queries")

    print("Initializing retrievers for label expansion...")
    embedder = EmbeddingModel(settings)
    semantic = SemanticRetriever(settings, embedder=embedder)
    bm25 = BM25Retriever.load(settings.bm25_index_path)
    hybrid = HybridRetriever(settings, embedder=embedder)
    reranker = CrossEncoderReranker(settings)
    CrossEncoderReranker.get_model(settings.reranker_model)

    expanded_dataset = []
    for index, entry in enumerate(seed_dataset, start=1):
        expanded = expand_entry(entry, semantic, bm25, hybrid, reranker)
        expanded_dataset.append(expanded)
        print(
            f"[{index}/{len(seed_dataset)}] {entry['query'][:50]}... "
            f"-> {len(expanded['relevant_chunk_ids'])} labels, "
            f"{len(expanded['relevant_arxiv_ids'])} papers"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in expanded_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    label_counts = [len(e["relevant_chunk_ids"]) for e in expanded_dataset]
    print(f"\nWrote expanded dataset to {output_path}")
    print(
        f"Label counts: min={min(label_counts)}, max={max(label_counts)}, "
        f"avg={sum(label_counts)/len(label_counts):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
