"""Retrieval evaluation metrics for gold-dataset benchmarking."""

from __future__ import annotations

from collections.abc import Iterable


def chunk_to_arxiv_id(chunk_id: str) -> str:
    """Extract arXiv ID from a chunk ID like ``2606.24133v1_chunk_0001``."""
    if "_chunk_" not in chunk_id:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")
    return chunk_id.rsplit("_chunk_", 1)[0]


def arxiv_ids_from_chunks(chunk_ids: Iterable[str]) -> list[str]:
    """Return unique arXiv IDs derived from chunk IDs, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk_id in chunk_ids:
        arxiv_id = chunk_to_arxiv_id(chunk_id)
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            ordered.append(arxiv_id)
    return ordered


def compute_precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """Chunk-level precision@k."""
    if not retrieved_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)
    return len(relevant_set & set(top_k)) / len(top_k)


def compute_recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """Chunk-level recall@k."""
    if not relevant_chunk_ids:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    return len(relevant_set & top_k) / len(relevant_set)


def compute_hit_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """1.0 if at least one relevant chunk appears in top-k, else 0.0."""
    if not relevant_chunk_ids or not retrieved_chunk_ids:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    return 1.0 if set(relevant_chunk_ids) & top_k else 0.0


def compute_mrr(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """Mean reciprocal rank for the first relevant chunk in top-k."""
    relevant_set = set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_paper_precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_arxiv_ids: list[str],
    k: int = 5,
) -> float:
    """Paper-level precision@k: fraction of top-k from a relevant paper."""
    if not retrieved_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    relevant_papers = set(relevant_arxiv_ids)
    hits = sum(1 for cid in top_k if chunk_to_arxiv_id(cid) in relevant_papers)
    return hits / len(top_k)


def compute_paper_hit_at_k(
    retrieved_chunk_ids: list[str],
    relevant_arxiv_ids: list[str],
    k: int = 5,
) -> float:
    """1.0 if at least one top-k chunk comes from a relevant paper."""
    if not retrieved_chunk_ids or not relevant_arxiv_ids:
        return 0.0
    relevant_papers = set(relevant_arxiv_ids)
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(chunk_to_arxiv_id(cid) in relevant_papers for cid in top_k) else 0.0


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    relevant_arxiv_ids: list[str] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Compute the full metric bundle for one query/strategy pair."""
    papers = relevant_arxiv_ids or arxiv_ids_from_chunks(relevant_chunk_ids)
    return {
        "precision_at_5": compute_precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        "recall_at_5": compute_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        "hit_at_5": compute_hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        "mrr": compute_mrr(retrieved_chunk_ids, relevant_chunk_ids, k),
        "paper_precision_at_5": compute_paper_precision_at_k(retrieved_chunk_ids, papers, k),
        "paper_hit_at_5": compute_paper_hit_at_k(retrieved_chunk_ids, papers, k),
    }
