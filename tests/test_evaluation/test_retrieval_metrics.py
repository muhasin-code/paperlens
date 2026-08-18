"""Tests for retrieval evaluation metrics."""

from src.paperlens.evaluation.metrics import (
    arxiv_ids_from_chunks,
    chunk_to_arxiv_id,
    compute_hit_at_k,
    compute_mrr,
    compute_paper_hit_at_k,
    compute_paper_precision_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)


class TestChunkToArxivId:
    def test_valid_chunk_id(self):
        assert chunk_to_arxiv_id("2606.24133v1_chunk_0001") == "2606.24133v1"

    def test_invalid_chunk_id(self):
        try:
            chunk_to_arxiv_id("invalid")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestArxivIdsFromChunks:
    def test_deduplicates_preserving_order(self):
        ids = arxiv_ids_from_chunks(
            [
                "2606.24133v1_chunk_0001",
                "2606.24133v1_chunk_0002",
                "2606.22984v2_chunk_0050",
            ]
        )
        assert ids == ["2606.24133v1", "2606.22984v2"]


class TestPrecisionRecallHit:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert compute_precision_at_k(retrieved, relevant, k=5) == 0.6
        assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0
        assert compute_hit_at_k(retrieved, relevant, k=5) == 1.0

    def test_zero_overlap(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert compute_precision_at_k(retrieved, relevant, k=3) == 0.0
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0
        assert compute_hit_at_k(retrieved, relevant, k=3) == 0.0

    def test_empty_retrieved(self):
        assert compute_precision_at_k([], ["a"], k=5) == 0.0
        assert compute_recall_at_k([], ["a"], k=5) == 0.0
        assert compute_hit_at_k([], ["a"], k=5) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert compute_mrr(["a", "b"], ["a"], k=5) == 1.0

    def test_third_rank(self):
        assert compute_mrr(["x", "y", "a"], ["a"], k=5) == 1 / 3

    def test_missing(self):
        assert compute_mrr(["x", "y"], ["a"], k=5) == 0.0


class TestPaperMetrics:
    def test_paper_precision(self):
        retrieved = [
            "2606.24133v1_chunk_0001",
            "2606.24133v1_chunk_0002",
            "2606.99999v1_chunk_0001",
        ]
        papers = ["2606.24133v1"]
        assert compute_paper_precision_at_k(retrieved, papers, k=3) == 2 / 3
        assert compute_paper_hit_at_k(retrieved, papers, k=3) == 1.0

    def test_paper_hit_miss(self):
        retrieved = ["2606.99999v1_chunk_0001"]
        papers = ["2606.24133v1"]
        assert compute_paper_hit_at_k(retrieved, papers, k=1) == 0.0


class TestRetrievalMetricsBundle:
    def test_returns_all_keys(self):
        metrics = compute_retrieval_metrics(
            retrieved_chunk_ids=["2606.24133v1_chunk_0001"],
            relevant_chunk_ids=["2606.24133v1_chunk_0001", "2606.24133v1_chunk_0002"],
            relevant_arxiv_ids=["2606.24133v1"],
            k=5,
        )
        assert set(metrics) == {
            "precision_at_5",
            "recall_at_5",
            "hit_at_5",
            "mrr",
            "paper_precision_at_5",
            "paper_hit_at_5",
        }
