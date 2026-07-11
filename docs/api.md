# PaperLens API Specification

This document describes the REST API served by the PaperLens FastAPI backend (Phase 1 / Milestone 1.4).

Base URL: `http://localhost:8000` (development)

---

## Endpoints

### `GET /health`

**Liveness / readiness probe.** Returns server status plus dependency health.

#### Response: `HealthResponse`

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` if Ollama reachable, `"degraded"` otherwise |
| `chroma_collection` | `string` | ChromaDB collection name (`"paperlens_chunks"`) |
| `chroma_vector_count` | `integer` | Number of vectors in the collection |
| `ollama_reachable` | `boolean` | Whether `GET /api/tags` succeeded (cached 10s) |
| `ollama_model` | `string` | Primary model from settings (`phi4-mini`) |
| `ollama_fallback_model` | `string` | Fallback model (`llama3.2:1b`) |

#### Example

```bash
curl -s http://localhost:8000/health | jq .
```

```json
{
  "status": "ok",
  "chroma_collection": "paperlens_chunks",
  "chroma_vector_count": 17330,
  "ollama_reachable": true,
  "ollama_model": "phi4-mini",
  "ollama_fallback_model": "llama3.2:1b"
}
```

---

### `POST /query`

**Main RAG endpoint.** Accepts a natural-language question, retrieves relevant chunks from the arXiv corpus, generates a cited answer via Ollama, and returns the answer with source citations.

#### Request: `QueryRequest`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Yes | — | Natural-language question (1–2000 chars) |
| `top_k` | `integer` | No | `settings.retrieval_top_k` (20) | Override retrieval depth (1–20) |
| `model` | `string` | No | `settings.ollama_model` | Override Ollama model name |

#### Response: `QueryResponse`

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | Generated answer text with inline `[chunk_id]` citations (model-dependent) |
| `citations` | `array[Citation]` | Source chunks used for the answer |
| `confidence` | `number` | Heuristic confidence (mean of top-3 retrieval scores, 0–1) |
| `retrieval_time_ms` | `number` | Query embedding + ChromaDB search latency |
| `generation_time_ms` | `number` | Ollama LLM generation latency |
| `total_time_ms` | `number` | End-to-end latency |

#### Citation Object

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `string` | Globally unique chunk ID (e.g. `2301.07597v2_chunk_0001`) |
| `arxiv_id` | `string` | Source paper arXiv ID with version |
| `title` | `string` | Source paper title |
| `authors` | `array[string]` | Source paper author list |
| `section_label` | `string` | Section this chunk belongs to (`abstract`, `introduction`, `method`, etc.) |
| `page_start` | `integer` | 1-indexed starting page |
| `page_end` | `integer` | 1-indexed ending page |
| `score` | `number` | Cosine similarity score (higher = more relevant) |
| `rank` | `integer` | 1-based rank in retrieved list |

#### Example Request

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do recent papers approach learning rate scheduling?", "top_k": 5}' | jq .
```

#### Example Request (with model override)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA?", "model": "llama3.2:1b"}' | jq .
```

#### Example Response

```json
{
  "answer": "Recent papers predominantly use cosine annealing with warm restarts [2606.24133v1_chunk_0003] and linear warmup followed by cosine decay [2401.12345v1_chunk_0012]. Some explore adaptive schedules based on gradient noise [2305.67890v2_chunk_0007].",
  "citations": [
    {
      "chunk_id": "2606.24133v1_chunk_0003",
      "arxiv_id": "2606.24133v1",
      "title": "Revisiting Learning Rate Schedules for Modern Architectures",
      "authors": ["Jane Doe", "John Smith"],
      "section_label": "method",
      "page_start": 4,
      "page_end": 5,
      "score": 0.7611,
      "rank": 1
    },
    {
      "chunk_id": "2401.12345v1_chunk_0012",
      "arxiv_id": "2401.12345v1",
      "title": "Efficient Training of Vision Transformers",
      "authors": ["Alice Chen"],
      "section_label": "experiments",
      "page_start": 8,
      "page_end": 9,
      "score": 0.7234,
      "rank": 2
    },
    {
      "chunk_id": "2305.67890v2_chunk_0007",
      "arxiv_id": "2305.67890v2",
      "title": "Adaptive Learning Rates via Gradient Noise Estimation",
      "authors": ["Bob Wilson", "Carol Brown"],
      "section_label": "method",
      "page_start": 3,
      "page_end": 4,
      "score": 0.6891,
      "rank": 3
    }
  ],
  "confidence": 0.7245,
  "retrieval_time_ms": 191.9,
  "generation_time_ms": 2847.3,
  "total_time_ms": 3039.2
}
```

> **Note on inline citations:** The prompt instructs the LLM to include `[chunk_id]` markers in the answer. Current CPU-only models (`phi4-mini`, `llama3.2:1b`) do not reliably follow this instruction. Phase 2.4 adds structured output validation and retry logic to enforce citation inclusion.

---

### `GET /`

Root endpoint — basic service info.

```bash
curl http://localhost:8000/
```

```json
{"name": "PaperLens", "version": "0.0.1", "docs": "/docs"}
```

---

## Interactive Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Expected Latency on CPU Hardware

> **Honest disclosure:** This system runs on a Dell Latitude 7490 (Intel i7-8650U, 8 GB RAM, no GPU). All inference is CPU-only.

| Stage | Typical Latency (ms) | Notes |
|---|---|---|
| Query embedding (BGE, CPU) | ~150–200 | `sentence-transformers` encode with prefix; cached after first request |
| ChromaDB HNSW search (top-20) | ~20–40 | In-memory index, cosine space |
| Context assembly | <5 | String concatenation |
| Ollama generation (`phi4-mini`, CPU) | **30,000–60,000** | Dominant latency; 3.8B params, CPU-only, 256 max tokens |
| Ollama generation (`llama3.2:1b`, CPU) | **15,000–30,000** | Fallback model, faster but lower quality |
| **Total (primary model)** | **~30,000–60,000** | End-to-end |
| **Total (fallback model)** | **~15,000–30,000** | End-to-end |

**Mitigations (future phases):**
- Phase 4.2: SSE streaming returns tokens incrementally (improves perceived latency via time-to-first-token)
- Deployment: GPU-enabled instance reduces generation to ~200–500 ms
- Phase 2: Hybrid retrieval + reranking improves answer quality at same generation cost

---

## Error Responses

| Status | Condition | Response Body |
|---|---|---|
| `422` | Request validation failed (empty query, `top_k` out of range) | `{ "detail": [...] }` (FastAPI default) |
| `502` | Both primary and fallback Ollama models failed | `{ "detail": "Both primary (phi4-mini) and fallback (llama3.2:1b) models failed" }` |
| `500` | Unexpected server error | `{ "detail": "Internal server error" }` |

---

## Running the API Locally

```bash
# 1. Start Ollama (separate terminal)
ollama serve
ollama pull phi4-mini
ollama pull llama3.2:1b

# 2. Start PaperLens API
make run
# → uvicorn src.paperlens.main:app --reload --host 0.0.0.0 --port 8000

# 3. Test health
curl http://localhost:8000/health

# 4. Query (use fallback model for faster CPU responses)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is LoRA?", "model": "llama3.2:1b"}'

# 5. CLI smoke test
python scripts/query.py --host localhost "What is LoRA?"
```

---

## Source Files

| File | Purpose |
|---|---|
| `src/paperlens/main.py` | FastAPI app factory, lifespan (warms embedding model), CORS, router inclusion |
| `src/paperlens/api/routes.py` | Route handlers (`/query`, `/health`) |
| `src/paperlens/api/schemas.py` | Pydantic request/response models (`QueryRequest`, `Citation`, `QueryResponse`, `HealthResponse`) |
| `src/paperlens/api/rag.py` | `RAGService`: retrieval → context assembly → Ollama generation with fallback → citations |
| `src/paperlens/embedding/retriever.py` | `SemanticRetriever`: query embed → search → reconstruct `Chunk` (accepts cached embedder) |
| `src/paperlens/embedding/embedder.py` | `EmbeddingModel`: wraps `sentence-transformers`, BGE query prefix, normalization (singleton cached) |
| `scripts/query.py` | CLI smoke-test client (defaults to `localhost`) |
| `tests/test_api/test_routes.py` | Route-level tests (mocked Ollama + retriever) |
| `tests/test_api/test_rag.py` | Service-level tests (mocked Ollama + retriever) |
| `docs/api.md` | This file |
