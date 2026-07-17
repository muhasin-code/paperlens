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
