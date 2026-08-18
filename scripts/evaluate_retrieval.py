#!/usr/bin/env python3
"""Retrieval quality benchmark across four retrieval strategies.

Strategies:
1. Baseline semantic retrieval (SemanticRetriever, top-5)
2. Hybrid retrieval with RRF (HybridRetriever, top-5)
3. Hybrid + cross-encoder reranking (Hybrid top-20 -> rerank to 5)
4. Semantic + cross-encoder reranking (Semantic top-50 -> rerank to 5)

Runs in HEADLESS mode (no Ollama/LLM required).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import SemanticRetriever
from src.paperlens.evaluation.metrics import compute_retrieval_metrics
from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import get_settings

STRATEGIES = ("baseline", "hybrid", "hybrid_reranker", "semantic_reranker")
METRIC_NAMES = (
    "precision_at_5",
    "recall_at_5",
    "hit_at_5",
    "mrr",
    "paper_precision_at_5",
    "paper_hit_at_5",
)
SEMANTIC_RERANK_CANDIDATES = 50
HYBRID_RERANK_CANDIDATES = 20
EVAL_TOP_K = 5


def load_gold_dataset(path: Path) -> list[dict]:
    """Load gold evaluation dataset from JSONL file."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _strategy_result(
    chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    relevant_arxiv_ids: list[str],
) -> dict[str, Any]:
    metrics = compute_retrieval_metrics(
        retrieved_chunk_ids=chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        relevant_arxiv_ids=relevant_arxiv_ids,
        k=EVAL_TOP_K,
    )
    return {"chunk_ids": chunk_ids, **metrics}


def evaluate_query(
    query: str,
    relevant_chunk_ids: list[str],
    relevant_arxiv_ids: list[str],
    semantic_retriever: SemanticRetriever,
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> dict[str, Any]:
    """Evaluate all four strategies for a single query."""
    results: dict[str, Any] = {
        "query": query,
        "relevant_chunk_ids": relevant_chunk_ids,
        "relevant_arxiv_ids": relevant_arxiv_ids,
    }

    semantic_top5 = semantic_retriever.search(query=query, top_k=EVAL_TOP_K)
    results["baseline"] = _strategy_result(
        [r.chunk.chunk_id for r in semantic_top5],
        relevant_chunk_ids,
        relevant_arxiv_ids,
    )

    hybrid_top5 = hybrid_retriever.search(query=query, top_k=EVAL_TOP_K)
    results["hybrid"] = _strategy_result(
        [r.chunk.chunk_id for r in hybrid_top5],
        relevant_chunk_ids,
        relevant_arxiv_ids,
    )

    hybrid_candidates = hybrid_retriever.search(query=query, top_k=HYBRID_RERANK_CANDIDATES)
    hybrid_reranked = reranker.rerank(query=query, results=hybrid_candidates, top_k=EVAL_TOP_K)
    results["hybrid_reranker"] = _strategy_result(
        [r.chunk.chunk_id for r in hybrid_reranked],
        relevant_chunk_ids,
        relevant_arxiv_ids,
    )

    semantic_candidates = semantic_retriever.search(query=query, top_k=SEMANTIC_RERANK_CANDIDATES)
    semantic_reranked = reranker.rerank(query=query, results=semantic_candidates, top_k=EVAL_TOP_K)
    results["semantic_reranker"] = _strategy_result(
        [r.chunk.chunk_id for r in semantic_reranked],
        relevant_chunk_ids,
        relevant_arxiv_ids,
    )

    return results


def _pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100


def _format_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _average_metric(results: list[dict], strategy: str, metric: str) -> float:
    return sum(r[strategy][metric] for r in results) / len(results)


def _build_stats(results: list[dict], dataset_size: int) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "dataset_size": dataset_size,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategies": list(STRATEGIES),
        "metrics": list(METRIC_NAMES),
    }

    for metric in METRIC_NAMES:
        for strategy in STRATEGIES:
            stats[f"{strategy}_{metric}"] = _average_metric(results, strategy, metric)

    baseline_p5 = stats["baseline_precision_at_5"]
    for strategy in STRATEGIES[1:]:
        strategy_p5 = stats[f"{strategy}_precision_at_5"]
        stats[f"{strategy}_precision_delta_pct"] = _pct_change(strategy_p5, baseline_p5)

    return stats


def _metric_table_row(stats: dict[str, Any], metric: str) -> str:
    cells = [metric]
    baseline = stats[f"baseline_{metric}"]
    cells.append(f"{baseline:.3f}")
    for strategy in STRATEGIES[1:]:
        value = stats[f"{strategy}_{metric}"]
        delta = _pct_change(value, baseline)
        cells.append(f"{value:.3f} ({_format_pct(delta)})")
    return "| " + " | ".join(cells) + " |"


def _analysis_sentence(name: str, delta_pct: float, metric: str) -> str:
    direction = "improves" if delta_pct > 0 else "reduces" if delta_pct < 0 else "does not change"
    return (
        f"**{name}** {direction} average {metric} by "
        f"**{_format_pct(delta_pct)}** relative to baseline semantic retrieval."
    )


def generate_markdown_report(results: list[dict], stats: dict[str, Any]) -> str:
    """Generate markdown report with comparison tables and conditional analysis."""
    lines = [
        "# Retrieval Quality Benchmark",
        "",
        "## Evaluation Setup",
        "",
        f"- **Dataset size**: {stats['dataset_size']} queries",
        "- **Retrieval strategies**: Baseline Semantic, Hybrid RRF, Hybrid + Reranker, Semantic + Reranker",
        "- **Metrics**: Precision@5, Recall@5, Hit@5, MRR, Paper-Precision@5, Paper-Hit@5",
        "- **Environment**: CPU-only (Dell Latitude 7490, Intel UHD 620, 7.6 GB RAM)",
        "- **Mode**: Headless (no Ollama/LLM dependencies)",
        "",
        "## Comparison Tables",
        "",
        "### Precision and Recall (chunk-level)",
        "",
        "| Metric | Baseline | Hybrid (RRF) | Hybrid + Reranker | Semantic + Reranker |",
        "|--------|----------|--------------|-------------------|---------------------|",
        _metric_table_row(stats, "precision_at_5"),
        _metric_table_row(stats, "recall_at_5"),
        _metric_table_row(stats, "hit_at_5"),
        _metric_table_row(stats, "mrr"),
        "",
        "### Paper-level metrics",
        "",
        "| Metric | Baseline | Hybrid (RRF) | Hybrid + Reranker | Semantic + Reranker |",
        "|--------|----------|--------------|-------------------|---------------------|",
        _metric_table_row(stats, "paper_precision_at_5"),
        _metric_table_row(stats, "paper_hit_at_5"),
        "",
        "## Methodology",
        "",
        "### Strategies Evaluated",
        "",
        "1. **Baseline Semantic**: `SemanticRetriever` top-5 cosine search over ChromaDB.",
        "2. **Hybrid (RRF)**: `HybridRetriever` merges semantic + BM25 with Reciprocal Rank Fusion.",
        "3. **Hybrid + Reranker**: Hybrid top-20 candidates reranked to top-5.",
        f"4. **Semantic + Reranker**: Semantic top-{SEMANTIC_RERANK_CANDIDATES} candidates reranked to top-5.",
        "",
        "### Gold Dataset",
        "",
        "Each query includes `relevant_chunk_ids` (5-10 manually curated labels) and "
        "`relevant_arxiv_ids` for paper-level scoring. Labels were expanded using both "
        "semantic and BM25 discovery from seed papers to reduce single-retriever bias.",
        "",
        "### Metric Definitions",
        "",
        "- **Precision@5**: |relevant chunks ∩ top-5| / 5",
        "- **Recall@5**: |relevant chunks ∩ top-5| / |all relevant chunks|",
        "- **Hit@5**: 1 if any relevant chunk appears in top-5, else 0",
        "- **MRR**: reciprocal rank of the first relevant chunk in top-5",
        "- **Paper-Precision@5**: fraction of top-5 chunks from a relevant paper",
        "- **Paper-Hit@5**: 1 if any top-5 chunk comes from a relevant paper",
        "",
        "## Detailed Results",
        "",
    ]

    for result in results:
        lines.append(f"### Query: {result['query'][:60]}...")
        lines.append("")
        lines.append(f"- **Relevant papers**: {', '.join(result['relevant_arxiv_ids'])}")
        lines.append(f"- **Relevant chunks**: {', '.join(result['relevant_chunk_ids'])}")
        for strategy in STRATEGIES:
            metrics = result[strategy]
            lines.append(
                f"- **{strategy}**: P@5={metrics['precision_at_5']:.3f}, "
                f"R@5={metrics['recall_at_5']:.3f}, Hit@5={metrics['hit_at_5']:.0f}, "
                f"MRR={metrics['mrr']:.3f}, Paper-P@5={metrics['paper_precision_at_5']:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Analysis",
            "",
            "### Performance Summary",
            "",
            _analysis_sentence(
                "Hybrid (RRF)",
                stats["hybrid_precision_delta_pct"],
                "precision@5",
            ),
            "",
            _analysis_sentence(
                "Hybrid + Reranker",
                stats["hybrid_reranker_precision_delta_pct"],
                "precision@5",
            ),
            "",
            _analysis_sentence(
                "Semantic + Reranker",
                stats["semantic_reranker_precision_delta_pct"],
                "precision@5",
            ),
            "",
            "Recall@5 and paper-level metrics should be used alongside precision@5 when "
            "judging hybrid retrieval, since BM25 can surface relevant papers with different "
            "chunk boundaries than the gold labels.",
            "",
            "## Limitations",
            "",
            "1. **Small test set**: ~25 queries provide directional signal, not exhaustive coverage.",
            "2. **Static corpus**: Results depend on the current arXiv snapshot.",
            "3. **Chunk-level labels**: Adjacent chunks from the same paper may be equally valid.",
            "4. **CPU-only latency**: Latency numbers are indicative, not benchmarked here.",
            "",
            "---",
            "",
            "## Files",
            "",
            "- `eval/gold_dataset.jsonl`: Gold evaluation dataset",
            "- `scripts/validate_gold_dataset.py`: Validates labels against ChromaDB",
            "- `scripts/evaluate_retrieval.py`: Headless evaluation script",
            "",
            f"*Benchmark generated: {stats['timestamp']}*",
            "",
        ]
    )

    return "\n".join(lines)


def _print_summary(stats: dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<22} {'P@5':>8} {'R@5':>8} {'Hit@5':>8} {'MRR':>8} {'Paper-P@5':>10}")
    print("-" * 70)
    for strategy in STRATEGIES:
        label = strategy.replace("_", " ").title()
        print(
            f"{label:<22} "
            f"{stats[f'{strategy}_precision_at_5']:>8.3f} "
            f"{stats[f'{strategy}_recall_at_5']:>8.3f} "
            f"{stats[f'{strategy}_hit_at_5']:>8.3f} "
            f"{stats[f'{strategy}_mrr']:>8.3f} "
            f"{stats[f'{strategy}_paper_precision_at_5']:>10.3f}"
        )
    print("=" * 70)
    print("Precision@5 delta vs baseline:")
    for strategy in STRATEGIES[1:]:
        delta = stats[f"{strategy}_precision_delta_pct"]
        print(f"  {strategy:<20} {_format_pct(delta)}")


def main() -> int:
    settings = get_settings()

    print(f"Loading gold evaluation dataset from {settings.gold_dataset_path}...")
    gold_dataset = load_gold_dataset(settings.gold_dataset_path)
    print(f"Loaded {len(gold_dataset)} queries")

    print("\nInitializing retrievers...")
    embedder = EmbeddingModel(settings)
    semantic_retriever = SemanticRetriever(settings, embedder=embedder)
    hybrid_retriever = HybridRetriever(settings, embedder=embedder)
    reranker = CrossEncoderReranker(settings)
    CrossEncoderReranker.get_model(settings.reranker_model)
    print("Retrievers initialized")

    print("\nRunning evaluation...")
    all_results = []
    for index, item in enumerate(gold_dataset, start=1):
        query = item["query"]
        print(f"Query {index}/{len(gold_dataset)}: {query[:50]}...")
        all_results.append(
            evaluate_query(
                query=query,
                relevant_chunk_ids=item["relevant_chunk_ids"],
                relevant_arxiv_ids=item.get("relevant_arxiv_ids") or [],
                semantic_retriever=semantic_retriever,
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
            )
        )

    stats = _build_stats(all_results, len(gold_dataset))

    md_path = Path("docs/evaluation/retrieval-metrics.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(generate_markdown_report(all_results, stats), encoding="utf-8")

    json_path = Path("eval/retrieval_benchmark_results.json")
    json_output = {
        "timestamp": stats["timestamp"],
        "dataset_size": stats["dataset_size"],
        "strategies": list(STRATEGIES),
        "metrics": list(METRIC_NAMES),
        "averages": {
            strategy: {metric: stats[f"{strategy}_{metric}"] for metric in METRIC_NAMES}
            for strategy in STRATEGIES
        },
        "precision_delta_pct_vs_baseline": {
            strategy: stats[f"{strategy}_precision_delta_pct"] for strategy in STRATEGIES[1:]
        },
        "detailed_results": all_results,
    }
    json_path.write_text(json.dumps(json_output, indent=2), encoding="utf-8")

    _print_summary(stats)
    print(f"\nResults written to:\n  - {md_path}\n  - {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
