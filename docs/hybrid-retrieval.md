# Hybrid Retrieval with Reciprocal Rank Fusion

This document describes the hybrid retrieval system implemented in Milestone 2.2, which combines semantic (dense) and BM25 (sparse) retrieval using Reciprocal Rank Fusion (RRF).

---

## Overview

Hybrid retrieval addresses the limitations of single-mode retrieval:
- **Semantic search** excels at capturing meaning but may miss exact keyword matches
- **BM25 keyword search** excels at exact term matching but cannot understand semantics

By running both retrievers in parallel and fusing their results, we achieve higher recall and more robust retrieval, especially for queries with technical terms or short keywords.

---

## Reciprocal Rank Fusion (RRF) Formula

The RRF algorithm assigns a score to each document d based on its rank in each retriever's result list:

```
score_RRF(d) = Σ_{i=1}^{n} 1 / (k + rank_i(d))
```

Where:
- n = number of retrievers (2: semantic and BM25)
- k = RRF constant (default 60)
- rank_i(d) = 1-based rank of document d in retriever i's result list

### Edge Case: Document in Only One Retriever

A chunk returned by only one retriever still receives an RRF score using its rank in that retriever's list. For example, if a chunk appears only in BM25 results at rank 3, its RRF score is `1 / (k + 3)`.

### Example Calculation

Consider a query "learning rate scheduling":

| chunk_id | Semantic Rank | BM25 Rank | RRF Score |
|----------|---------------|-----------|-----------|
| c1       | 1             | 2         | 1/61 + 1/62 = 0.0328 |
| c2       | 3             | 1         | 1/63 + 1/61 = 0.0326 |
| c3       | 5             | —         | 1/65 = 0.0154 |

Results are sorted by RRF score descending.

---

## Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `RRF_K` | 60 | The RRF constant k. Higher values reduce the impact of rank differences; lower values give more weight to top ranks. The value 60 is from the original RRF paper and works well for 2 retrievers. |
| `HYBRID_CANDIDATE_POOL` | 20 | Number of top candidates to retrieve from each retriever before RRF fusion. Larger pools increase recall but also latency. This is not the final result count; it's the pool size before reranking. |
| `RETRIEVAL_TOP_K` | 20 | Final number of results to return after RRF fusion (before reranking in Phase 2.3). |

### Parameter Tuning

- **RRF_K**: For 2 retrievers, k=60 is a good default. If you add more retrievers (e.g., dense retrieval variants), consider increasing k proportionally.
- **HYBRID_CANDIDATE_POOL**: Start with 20. If recall is low (valid chunks missing), increase to 50. Monitor latency impact.

---

## Implementation

### Class: `HybridRetriever`

Location: `src/paperlens/retrieval/hybrid.py`

```python
class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        semantic_retriever: SemanticRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        # Lazy initialization; retrievers created on first search
```

### Execution Flow

```
User query
    ↓
HybridRetriever.search()
    ↓
Parallel execution:
  ├─ SemanticRetriever.search(query, top_k=20) → dense results
  └─ BM25Retriever.search(query, top_k=20) → sparse results
    ↓
RRF fusion:
  ├─ Collect all chunk_ids from both results
  ├─ Compute RRF score for each
  └─ Sort by score descending
    ↓
Return top_k RetrievalResult objects
```

### CPU-Only Implementation Note

On the Dell Latitude 7490 (Intel UHD 620, no GPU), "parallelism" means Python-level sequential execution:
1. Run semantic retrieval (embed query + ChromaDB search)
2. Run BM25 retrieval (tokenize + BM25 score lookup)
3. Fuse results in memory

This is intentionally not GPU-parallel. The latency trade-off is acceptable because:
- BM25 retrieval is very fast (~5-10 ms for 17k chunks)
- Semantic retrieval dominates latency (~150-200 ms)
- RRF fusion is O(n) in the candidate pool

---

## Design Trade-offs: Hybrid vs Pure Semantic

### Quality Benefits

| Aspect | Semantic Only | Hybrid (Semantic + BM25) |
|---|---|---|
| Keyword matching | May miss exact terms | Exact terms captured by BM25 |
| Technical terms | Relies on embedding context | BM25 matches exact tokenization |
| Short queries | May under-retrieve | BM25 provides keyword signal |
| Long queries | Good coverage | Good coverage |
| Out-of-vocabulary | May work via context | May fail (tokenization mismatch) |

### Cost Considerations (CPU-only)

| Stage | Latency (ms) | Notes |
|---|---|---|
| Semantic retrieval | ~150-200 | Query embedding + ChromaDB search |
| BM25 retrieval | ~5-10 | Tokenization + score lookup |
| RRF fusion | <1 | In-memory dictionary operations |
| **Total hybrid** | ~160-220 | Semantic dominates; BM25 adds minimal overhead |

The hybrid approach adds ~10-20 ms of latency on CPU, which is negligible compared to the semantic retrieval baseline.

---

## Debug Endpoint: POST /retrieval/hybrid

Inspect hybrid retrieval results without running the full RAG pipeline:

```bash
curl -s -X POST http://localhost:8000/retrieval/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "learning rate scheduling", "top_k": 5}' | jq .
```

**Request:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Yes | — | Natural-language query (1–2000 chars) |
| `top_k` | `integer` | No | `5` | Number of results (1–50) |

**Response:** Array of `RetrievalResult` objects with `chunk_id`, `score` (RRF score), `rank`, `chunk` (full metadata).

---

## Source Files

| File | Purpose |
|---|---|
| `src/paperlens/retrieval/hybrid.py` | `HybridRetriever` class with RRF fusion |
| `src/paperlens/retrieval/__init__.py` | Package init, exports `HybridRetriever` |
| `src/paperlens/settings.py` | `rrf_k` and `hybrid_candidate_pool` config |
| `.env.example` | Environment variable placeholders |
| `src/paperlens/api/rag.py` | `RAGService` uses `HybridRetriever` |
| `src/paperlens/api/retrieval_routes.py` | `/retrieval/hybrid` debug endpoint |
| `tests/test_retrieval/test_hybrid.py` | Unit tests for RRF logic |
| `Makefile` | `hybrid-verify` target |

---

## Known Limitations

- **Sequential execution:** No true parallelism on CPU; retrievers run sequentially
- **Memory overhead:** Both indexes loaded simultaneously (~400 MB combined)
- **No query expansion:** Short queries may still under-retrieve; cross-encoder reranking (Phase 2.3) addresses this
- **Fixed k value:** RRF_K is constant; adaptive k based on query length could improve results

---

*Document created: 2026-07-23*
