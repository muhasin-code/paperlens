# PaperLens — Architecture Overview

This document provides a narrative description of the PaperLens system architecture.
For visual representations, see the linked diagrams at the end of this page.

---

## System Purpose

PaperLens is a research intelligence platform that ingests ML/AI papers from arXiv,
indexes them for retrieval, and answers natural language queries with cited, grounded responses.
A secondary capability — structured paper intelligence extraction — is powered by a fine-tuned
model. The system is designed to run entirely on open-source tooling with no paid API
dependencies.

---

## Component Overview

### Ingestion Layer (Phase 1)

Papers are fetched from arXiv using the arXiv Python API. PDFs are parsed with PyMuPDF
into section-aware chunks (600 tokens, 100-token overlap). Each chunk carries metadata:
paper ID, title, authors, section label, and position.

### Retrieval Layer (Phase 1–2)

Two parallel indexes power retrieval:

- **Semantic index** (ChromaDB): chunks embedded with `BAAI/bge-large-en-v1.5` via
  sentence-transformers.
- **Keyword index** (BM25): sparse retrieval over the same chunk corpus using `rank_bm25`.

At query time, both indexes return candidate lists. Reciprocal Rank Fusion (RRF) merges
them, and `cross-encoder/ms-marco-MiniLM-L-12-v2` reranks the top 20 candidates to
produce the final top 5 context chunks.

### Generation Layer (Phase 1)

Ollama serves `phi4-mini` locally (CPU inference on the development machine; larger models
on deployment). A versioned YAML prompt in `configs/prompts/` enforces grounded answers —
the LLM must cite the source chunk or refuse to answer. Responses are validated with
Pydantic before being returned.

### Extraction Layer (Phase 3)

A QLoRA fine-tuned variant of `Qwen2.5-3B-Instruct` handles structured extraction queries
(research question, method, datasets, metrics). The query router in the FastAPI layer
classifies each incoming request and directs it to either the RAG path or the extraction
path.

### API Layer (Phase 1)

FastAPI serves the `/query` endpoint (Phase 1 blocking; Phase 4 SSE streaming). Pydantic
schemas enforce request/response contracts. All inference calls pass through Langfuse
instrumentation (Phase 4) for tracing.

### Evaluation & CI (Phase 4)

A gold dataset of 75 curated QA pairs lives in `eval/gold_dataset.jsonl`. GitHub Actions
runs Ragas (faithfulness, answer relevancy, context precision) against this dataset on
every PR. A faithfulness score below 0.75 fails the build.

---

## Request Lifecycle (simplified)

```
User query
  → Gradio UI (Phase 1)
  → FastAPI /query
  → QueryRouter
       ├─ Conversational → HybridRetrieval (BM25 + semantic) → RRF → Reranker → Ollama → CitationValidator → SSEStream
       └─ Structured     → FineTunedExtractor → JSONOutput → SSEStream
```

---

## Technology Choices

| Layer | Tool | Rationale |
|---|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` | MTEB-leading open model; runs on CPU |
| Vector store | ChromaDB | Embedded, no separate service in Phase 1 |
| Keyword search | rank_bm25 | Minimal dependency; complements dense retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Strong precision gains at low latency |
| LLM | Ollama / phi4-mini | Local, zero API cost, swappable |
| Fine-tuning | QLoRA (TRL + PEFT) | VRAM-efficient; Colab-compatible |
| Observability | Langfuse (self-hosted) | Open-source; Docker-native |
| Evaluation | Ragas | Standard RAG metrics; CI-integrable |

---

## Architectural Diagrams

Full component-interaction and data-flow diagrams are in the `plan/` directory:

- [System Architecture (draw.io XML)](../plan/paperlens_architecture.xml)
- [Data Flowchart (draw.io XML)](../plan/paperlens_flowchart.xml)

To view: open either file at [app.diagrams.net](https://app.diagrams.net) using
File → Import From → Device.
