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
