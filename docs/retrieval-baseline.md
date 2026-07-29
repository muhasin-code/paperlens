# Retrieval Baseline — Semantic Index (Phase 1)

This document records the semantic retrieval baseline established at the end of Milestone 1.3. It captures the embedding model choice, index construction, performance metrics, and known limitations so future phases (hybrid retrieval, reranking, evaluation) can measure against a fixed reference.

---

## System Purpose

The semantic index provides the dense-retrieval half of PaperLens's Phase 1 retrieval layer. Given a natural-language query, it returns the top-k most semantically similar chunks from the arXiv corpus, each carrying full provenance (`chunk_id`, `arxiv_id`, `section_label`, page range) for downstream citation enforcement.

---

## Embedding Model

| Field | Value |
|---|---|
| **Model** | `BAAI/bge-large-en-v1.5` |
| **Dimension** | 1024 |
| **Normalization** | Unit-length (dot product == cosine similarity) |
| **Query prefix** | Applied to queries only: `Represent this sentence for searching relevant passages:` |
| **Device** | CPU (local); CUDA when available on cloud runners |
| **Framework** | `sentence-transformers >= 3.1.0` |
| **Batch size (corpus)** | 16 (memory-safe on 8 GB RAM) |

### Rationale

`BAAI/bge-large-en-v1.5` was selected because:

1. **MTEB-leading open model** — consistently ranks top-3 on English retrieval tasks without API keys.
2. **CPU-compatible** — 335 M parameters (~1.3 GB FP32) fits comfortably in 8 GB system RAM with headroom for OS and ChromaDB.
3. **Query-prefix convention** — BGE models are trained with an asymmetric query/document format; applying the prefix only to queries is essential for recall. The wrapper in `src/paperlens/embedding/embedder.py` enforces this automatically.
4. **Open weights** — no licensing friction for commercial or academic deployment.

---

## Index Parameters

| Field | Value |
|---|---|
| **Collection name** | `paperlens_chunks` |
| **Distance space** | `cosine` (ChromaDB HNSW with cosine space) |
| **Persistence path** | `./data/chroma/` |
| **Upsert batch size** | 5000 (stays under ChromaDB's internal limit of ~5461) |
| **Idempotency** | Upsert by `chunk_id` — safe to re-run; `--rebuild` resets collection |
| **Checkpoint file** | `./data/processed/embedding_checkpoint.jsonl` (resumable runs) |
| **Metadata fields** | `arxiv_id`, `title`, `authors` (semicolon-joined), `section_label`, `chunk_index`, `page_start`, `page_end`, `token_count`, `total_chunks_in_section`, `total_chunks_in_paper`, `dimension` |

### Metadata Design

ChromaDB requires metadata values to be str/int/float/bool. `authors` is a `list[str]` in the `Chunk` model; it is flattened to a semicolon-delimited string in `VectorStore._metadata_for()` so it survives round-trips. The retriever (`SemanticRetriever._chunk_from_metadata()`) reconstructs the list on read.

---

## Index Size

| Metric | Value |
|---|---|
| **Chunks in Parquet** | 17,330 |
| **Vectors in ChromaDB** | 17,330 |
| **Disk usage** | 381 MB |
| **Source file** | `data/processed/chunks.parquet` |
| **Avg chunk length** | ~2,050 characters (~600 tokens) |
| **Embedding model size (FP32)** | ~1.3 GB |

---

## Retrieval Latency Baseline

Measured on local CPU box (Dell Latitude 7490, Intel i7-8650U, 8 GB RAM, no GPU).

| Metric | Value |
|---|---|
| **Query** | `How do recent papers approach learning rate scheduling?` |
| **Top-k** | 5 |
| **Warm-up run** | 1 (discarded) |
| **Measured run** | 1 |
| **Retrieval latency** | 191.9 ms (query embed + top-5 search) |
| **Top-1 section** | `abstract` |
| **Top-1 arxiv_id** | `2606.24133v1` |
| **Top-1 score** | 0.7611 |

> **Note:** Latency includes query embedding (BGE prefix + encode) + ChromaDB HNSW search + metadata reconstruction. This is the Phase 1 baseline; Phase 2 will add BM25 hybrid search and cross-encoder reranking on top.

---

## Rebuild Instructions

```bash
# Clean index and re-embed from scratch
python scripts/embed.py --rebuild

# Or via Make
make embed

# Dry-run (smoke test, no writes)
python scripts/embed.py --dry-run --limit 5
```

The pipeline is fully resumable. If interrupted, re-run `make embed` — already-embedded chunk IDs are read from `embedding_checkpoint.jsonl` and skipped.

---

## Sample Query + Top Result

**Query:** `How do recent papers approach learning rate scheduling?`

**Top result:**
- **Section:** `abstract`
- **Paper:** `2606.24133v1`
- **Cosine similarity score:** `0.7611`
- **Section label:** `abstract` (chunk from the paper's abstract section)

---

## Pipeline Flow (Query Time)

```
User query
    ↓
EmbeddingModel.embed_query()  # adds BGE prefix, normalizes
    ↓
VectorStore.query()           # ChromaDB HNSW search (cosine)
    ↓
SemanticRetriever._chunk_from_metadata()  # reconstruct Chunk objects
    ↓
RetrievalResult[]             # ranked by score = 1 - cosine_distance
```

---

## Known Limitations

- **CPU-only query embedding** — local development runs on CPU; latency is ~190 ms. Cloud deployment with GPU will be significantly faster.
- **Single dense vector per chunk** — no multi-vector (ColBERT) or late-interaction representations yet.
- **No query expansion** — short queries may under-retrieve; Phase 2 adds BM25 hybrid + RRF.
- **No cross-encoder reranking** — top-5 from dense search are passed directly to the LLM. Phase 2 introduces `cross-encoder/ms-marco-MiniLM-L-12-v2` reranking.
- **Static index** — index is rebuilt only via `--rebuild`. No incremental updates for new arXiv papers yet.
- **Section-awareness is metadata-only** — retrieval does not yet boost by section type (e.g., prefer `method` over `related-work`).
- **No evaluation harness** — this document is the only quantitative record. Phase 4 introduces Ragas on a gold dataset (`eval/gold_dataset.jsonl`).

---

## Source Files

| File | Purpose |
|---|---|
| `src/paperlens/embedding/embedder.py` | `EmbeddingModel`: wraps `sentence-transformers`, applies BGE query prefix, normalizes |
| `src/paperlens/embedding/vector_store.py` | `VectorStore`: ChromaDB persistent client, batched upsert, cosine search |
| `src/paperlens/embedding/retriever.py` | `SemanticRetriever`: query embed → search → reconstruct `Chunk` |
| `src/paperlens/embedding/pipeline.py` | `EmbeddingPipeline`: load Parquet → embed in batches → upsert with checkpointing |
| `src/paperlens/embedding/models.py` | `EmbeddedChunk`, `RetrievalResult` Pydantic models |
| `scripts/embed.py` | CLI entry point (`--dry-run`, `--limit`, `--force`, `--rebuild`) |
| `tests/test_embedding/test_embedder.py` | Unit tests (mocked `SentenceTransformer`) |
| `tests/test_embedding/test_vector_store.py` | Unit tests (temp ChromaDB directory) |

---

## Next Steps (Phase 2)

| Item | Target | Owner |
|---|---|---|
| BM25 index build (`rank_bm25`) | `docs/bm25-baseline.md` | Phase 2.1 |
| Hybrid search (RRF merge) | `docs/hybrid-retrieval.md` | Phase 2.2 |
| Cross-encoder reranker | `docs/reranker-baseline.md` | Phase 2.3 |
| Retrieval evaluation (Ragas) | `eval/gold_dataset.jsonl` | Phase 4 |

---

*Baseline recorded: 2026-07-10*

---

# BM25 Keyword Index (Phase 2.1)

This section records the BM25 sparse retrieval index built as part of Milestone 2.1. It complements the semantic index documented above.

## Model and Parameters

| Field | Value |
|---|---|
| **Library** | `rank_bm25 >= 0.2.2` (`BM25Okapi` variant) |
| **k1** | 1.5 (term frequency saturation) |
| **b** | 0.75 (document length normalization) |
| **Tokenization** | Whitespace split + lowercase (no stemming; CPU-fast) |
| **Persistence** | `joblib.dump` (pickle-based serialization) to `data/bm25_index.pkl` |
| **Index path (settings)** | `settings.bm25_index_path` → `./data/bm25_index.pkl` |

## Index Statistics

| Metric | Value |
|---|---|
| **Corpus size** | 17,330 chunks (from `data/processed/chunks.parquet`) |
| **Vocabulary size** | _fill_ unique terms |
| **Build time** | _fill_ ms |
| **Index file size** | 112.1 MB (117,540,215 bytes) |
| **Index path** | `data/bm25_index.pkl` |

> **Fill in the values above** from the output of `make bm25-build` (run with `--rebuild` to see stats).

## Rebuild Instructions

```bash
# Full rebuild (deletes existing index, rebuilds from chunks.parquet)
make bm25-build BM25_REBUILD=1

# Or directly:
python scripts/build_bm25.py --rebuild

# Quick build (skips if index already exists)
make bm25-build

# Verify the index is functional
make bm25-verify

# Remove the index
make bm25-clean
```

The index is fully deterministic given the same `chunks.parquet` and parameters. Rebuilding after adding new papers to the corpus is the supported workflow for incremental updates (full rebuild required; no partial upsert mechanism exists yet).

## Standalone Debug Endpoint

A standalone debug endpoint is available for inspecting BM25 results without running the full RAG pipeline:

```
POST /retrieval/bm25
```

```bash
curl -s -X POST http://localhost:8000/retrieval/bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "learning rate scheduling", "top_k": 5}' | jq .
```

Request:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Yes | — | Natural-language query (1–2000 chars) |
| `top_k` | `integer` | No | `5` | Number of results (1–50) |

Response: array of `BM25Result` objects with `chunk_id`, `score`, `rank`, `arxiv_id`, `title`, `authors`, `section_label`, `page_start`, `page_end`, `text`.

Returns `503 Service Unavailable` if the index has not been built yet (run `make bm25-build`).

## Source Files

| File | Purpose |
|---|---|
| `src/paperlens/retrieval/bm25.py` | `BM25Retriever` and `BM25Result` |
| `src/paperlens/retrieval/__init__.py` | Package init, exports `BM25Retriever` and `BM25Result` |
| `src/paperlens/api/retrieval_routes.py` | `/retrieval/bm25` debug endpoint |
| `src/paperlens/api/schemas.py` | `Bm25SearchRequest` schema |
| `scripts/build_bm25.py` | CLI build script (`--rebuild` flag) |
| `tests/test_retrieval/test_bm25.py` | 25 unit tests (tmp_path, no I/O) |

## Known Limitations

- **No incremental updates.** New chunks require a full rebuild. No upsert-by-chunk-id mechanism exists yet.
- **Simple tokenization.** Whitespace + lowercase only — no Porter stemming or lemmatization. Intentional for CPU speed and to avoid over-normalizing technical terms (e.g., "Transformer" vs "transformer").
- **Static parameters.** k1 and b are fixed at 1.5 and 0.75 (BM25Okapi defaults). Parameter tuning is deferred to Phase 2.2 (RRF fusion evaluation).
- **No query expansion.** Short or ambiguous queries may under-retrieve. Phase 2.2 RRF fusion with semantic search addresses this.
- **Memory-bound.** The full index (~17k chunks, tokenized corpus, BM25Okapi object) must fit in RAM. On the 7.6 GB target machine, this is comfortably under 500 MB.

---

## Cross-Encoder Reranking (Phase 2.3)

This section documents the cross-encoder reranking system that refines hybrid retrieval results.

### Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| **RERANKER_MODEL** | `cross-encoder/ms-marco-MiniLM-L-12-v2` | HuggingFace model identifier |
| **RERANK_TOP_K** | 5 | Number of top candidates passed to LLM after reranking |

### Retriever vs Reranker: Why Two Stages?

| Aspect | Bi-Encoder Retriever | Cross-Encoder Reranker |
|---|---|---|
| Speed | Fast (~200-300 ms) | Slower (~200-500 ms for 20 chunks) |
| Accuracy | Good | Higher precision |
| Use case | First-stage recall | Second-stage refinement |
| Model size | ~1.3 GB | ~85 MB |

### Reranking Flow

```
Query → HybridRetriever.search(top_k=20)
    ↓
CrossEncoderReranker.rerank(top_k=5)
    ↓
LLM prompt (top 5 chunks with cross-encoder scores)
```

### Latency Baseline (CPU-only)

| Operation | Latency (ms) |
|---|---|
| Rerank 20 candidates | 200-500 |
| Rerank 5 candidates | 50-150 |
| Model load (first run) | 1000-2000 (download) |
| Model load (cached) | 100-200 |

### Memory Impact

| Component | Memory |
|---|---|
| BM25 index | ~117 MB |
| ChromaDB + embeddings | ~400 MB |
| BGE embedding model | ~1.3 GB |
| Cross-encoder reranker | ~85 MB |
| **Total** | ~1.9 GB |

### Debug Endpoint: POST /retrieval/rerank

```bash
POST /retrieval/rerank

curl -s -X POST http://localhost:8000/retrieval/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "learning rate scheduling", "top_k": 5}' | jq .
```

### Source Files

| File | Purpose |
|---|---|
| `src/paperlens/retrieval/reranker.py` | `CrossEncoderReranker` class |
| `src/paperlens/api/rag.py` | `RAGService` applies reranker |
| `src/paperlens/api/retrieval_routes.py` | `/retrieval/rerank` endpoint |
| `tests/test_retrieval/test_reranker.py` | Unit tests |
| `scripts/verify_reranker.py` | Smoke test |

### Known Limitations

- CPU-only reranking adds 200-500 ms latency
- Model download (~85 MB) required on first run
- Cross-encoder processes pairs sequentially

*BM25 section added: 2026-07-23*
