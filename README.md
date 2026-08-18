# PaperLens

Status
License

Production-grade research intelligence over ML/AI arXiv papers — hybrid retrieval, fine-tuned structured extraction, Langfuse observability, and Ragas CI evaluation gating.

> **Note:** Phase 1 (Core RAG Pipeline) is complete.

## Architecture

Architecture
*Open in [draw.io](https://app.diagrams.net) via File → Import From → Device.*

## Phase 1 Demo

PaperLens Demo
*The React UI querying 500 cs.LG papers with cited answers and latency breakdown. API runs on FastAPI + Ollama (CPU).*

## Evaluation Results (Phase 1)



### Inference Benchmarks (CPU Only — Dell Latitude 7490, Intel i7-8650U, 8 GB RAM)


| Model                    | Tokens/sec | TTFT (ms) | Peak RSS (MB) | Total Latency (ms) |
| ------------------------ | ---------- | --------- | ------------- | ------------------ |
| `llama3.2:1b` (primary)  | 28.4       | 180       | 240           | 1,100              |
| `phi4-mini` (fallback)   | 11.8       | 420       | 512           | 2,800              |


> **Honest disclosure:** No GPU. All inference runs on CPU. `phi4-mini` ~3.8B params takes 2.5–5.5 s end-to-end; `llama3.2:1b` ~1.3B params takes 1–2 s. See [docs/benchmarks.md](docs/benchmarks.md) for methodology.



### Retrieval Baseline

- **Embedding model:** `BAAI/bge-large-en-v1.5` (1024-dim)
- **Vector store:** ChromaDB (HNSW, cosine)
- **Corpus:** 17,330 chunks from 499 papers (cs.LG, 2023–2024)
- **Query latency (embed + top-5 search):** ~192 ms

Full details: `[docs/retrieval-baseline.md](docs/retrieval-baseline.md)`


## Retrieval Benchmark (Phase 2)

This milestone validates retrieval quality across four strategies on a curated gold dataset. The key finding: **hybrid retrieval (RRF) significantly outperforms baseline semantic search**, while cross-encoder reranking with `ms-marco-MiniLM-L-12-v2` degrades quality on 600-token academic chunks and is disabled in production.

### Methodology

- **Dataset**: 26 manually curated queries with known relevant chunk IDs (5–15 labels each, expanded from both semantic and BM25 discovery)
- **Metrics**: Precision@5, Recall@5, Hit@5, MRR, Paper-Precision@5, Paper-Hit@5, Soft nDCG@5 (partial credit for same-paper chunks)
- **Environment**: CPU-only (Dell Latitude 7490, Intel UHD 620, 7.6 GB RAM)
- **Retrieval modes**: Baseline semantic → Hybrid (RRF) → Hybrid + Reranker → Semantic + Reranker
- **Gold set construction**: Pool-depth-corrected (labels pooled from top-50 of each retriever, interleaved to avoid semantic bias)

### Results

| Strategy | Precision@5 | Recall@5 | Hit@5 | MRR | Paper-P@5 | Soft nDCG@5 | Status |
|----------|-------------|----------|-------|-----|-----------|-------------|--------|
| Baseline Semantic | 0.423 | 0.212 | 0.885 | 0.683 | 0.592 | — | Baseline |
| **Hybrid (RRF)** | **0.515** | **0.258** | **0.962** | **0.749** | **0.692** | **0.623** | **Production** |
| Hybrid + Reranker | 0.308 | 0.154 | 0.846 | 0.608 | 0.469 | 0.406 | Disabled |
| Semantic + Reranker | 0.231 | 0.115 | 0.692 | 0.468 | 0.369 | — | Disabled |

**Hybrid RRF improves Precision@5 by +21.8% over baseline** (0.515 vs 0.423).

### Analysis

The **hybrid retrieval** approach delivers measurable improvements by combining the complementary strengths of dense semantic matching (captures meaning) and BM25 keyword search (captures exact terms). Weighted RRF (2:1 semantic:BM25) with a candidate pool of 50 yields the best trade-off.

The **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-12-v2`) was trained on 70–100 token web search passages. On PaperLens's 600-token academic chunks (dense with equations, citations, and notation), its learned relevance signal is **anti-correlated** with academic relevance. It actively demotes relevant chunks and promotes irrelevant ones, reducing Precision@5 by 27–46% relative to hybrid.

Per-query soft nDCG@5 (0.5 credit for same-paper chunks) confirms: hybrid scores 0.623, reranker 0.406. The reranker helps on definitional queries (e.g., "attention mechanisms") but catastrophically fails on notation-heavy queries (e.g., "batch normalization," "Adam vs SGD," "diffusion models").

### Performance Trade-offs

| Strategy | Precision@5 | Latency* | Memory* | Use Case |
|----------|-------------|----------|---------|----------|
| Baseline | 0.423 | ~150 ms | ~1.3 GB | Fast initial demo |
| **Hybrid (RRF)** | **0.515** | ~160 ms | ~1.9 GB | **Production standard** |
| Hybrid + Reranker | 0.308 | ~460 ms | ~2.2 GB | Disabled (degrades quality) |
| Semantic + Reranker | 0.231 | ~500 ms | ~2.0 GB | Disabled (degrades quality) |

*Latency/memory estimates from M2.5 benchmark environment (CPU-only, Dell Latitude 7490).

### Files

- `eval/gold_dataset.jsonl`: Gold evaluation dataset (26 queries, 5–15 chunk labels each, with `relevant_arxiv_ids` for paper-level metrics)
- `scripts/evaluate_retrieval.py`: Headless evaluation script (4 strategies, 7 metrics including soft nDCG)
- `scripts/discover_relevant_chunks.py`: Semantic + BM25 discovery for gold set construction
- `scripts/expand_gold_dataset.py`: Pool-depth-corrected label expansion (interleaved, top-50 pool)
- `scripts/validate_gold_dataset.py`: Validates all chunk IDs exist in ChromaDB
- `docs/evaluation/retrieval-metrics.md`: Full benchmark documentation with per-query soft nDCG breakdown
- `eval/retrieval_benchmark_results.json`: Structured results for programmatic access


## Quick Start

```bash
# 1. Clone and setup
git clone git@github.com:muhasin-code/paperlens.git
cd paperlens
python -m venv .venv && source .venv/bin/activate
make install-dev

# 2. Start Ollama (separate terminal)
ollama serve
ollama pull phi4-mini
ollama pull llama3.2:1b

# 3. Ingest papers (one-time, ~1-2 hours)
make ingest

# 4. Parse PDFs into chunks (~30-60 min)
make parse

# 5. Embed chunks into ChromaDB (~10-20 min)
make embed

# 6. Start API server
make run
# → http://localhost:8000/docs

# 7. Start React dev server (new terminal, optional for hot reload)
make frontend-dev
# → http://localhost:3000
```

Or production mode (serves React build from FastAPI):

```bash
make run-full
# → http://localhost:8000
```



## Live Demo

*Coming in Phase 6 (deployment).*

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
