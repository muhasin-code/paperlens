# PaperLens — Inference Benchmarks (CPU Only)

> **Hardware disclosure:** All benchmarks run on a **Dell Latitude 7490** (Intel i7-8650U, 4C/8T, 8 GB DDR4, **no GPU** — Intel UHD 620 only). Ollama runs CPU-only. These numbers represent the *real* latency a user would see on a typical laptop without cloud GPUs.

---

## Methodology

- **Models:** `phi4-mini` (1.3B params, primary)
- **Client:** `ollama` Python client v0.4+ with `stream=True`
- **Prompts:** 5 diverse ML/AI questions (see `scripts/benchmark_ollama.py::PROMPTS`)
- **Runs:** 3 runs per model per prompt (15 total); median reported
- **Metrics:**
  - **Tokens/sec:** estimated tokens generated per second (character count / 4)
  - **TTFT (ms):** time-to-first-token from stream start
  - **Peak RSS (MB):** process resident set size via `psutil.Process.memory_info().rss`
  - **Total latency (ms):** wall-clock from first request byte to stream end
- **Generation config:** `temperature=0.1`, `num_predict=256`
- **Warm-up:** one unmeasured generation per model before timed runs

---

## Results

| Model | Tokens/sec (median) | TTFT (ms, median) | Peak RSS (MB, median) | Total Latency (ms, median) |
|---|---:|---:|---:|---:|
| `phi4-mini` | 10.1 | 475.5 | 49.5 | 22,102 |

_Raw data: [`docs/benchmarks.json`](benchmarks.json)_

---

## Analysis

### `phi4-mini` (Primary)

- **Quality:** Adequate for simple factoid questions; weaker on complex synthesis compared to larger models
- **Speed:** ~10.1 tokens/sec; TTFT ~475 ms
- **Memory:** ~49.5 MB peak RSS — extremely lightweight for a 1.3B parameter model
- **Use case:** Primary model for PaperLens; optimal for CPU-only deployments where latency matters

### CPU Reality Check

| Stage | Typical Latency (ms) | Notes |
|---|---:|---|
| Query embedding (BGE-large, CPU) | 150–200 | `sentence-transformers` encode with prefix |
| ChromaDB HNSW search (top-20) | 20–40 | In-memory index, cosine space |
| Context assembly | <5 | String concatenation |
| **Ollama generation (`phi4-mini`)** | **~22,100** | **Dominant cost** — 1.3B params on CPU |
| **Total (primary)** | **~22,500** | End-to-end |

**Mitigations (future phases):**
- Phase 4.2: SSE streaming returns tokens incrementally (improves *perceived* latency via TTFT)
- Deployment: GPU-enabled instance reduces generation to ~200–500 ms
- Phase 2: Hybrid retrieval + reranking improves answer quality at same generation cost

---

## Per-Prompt Breakdown

| Prompt | Tokens/sec (median) | TTFT (ms, median) | Peak RSS (MB, median) |
|---|---:|---:|---:|
| What is LoRA and how does it work? | 9.1 | 481.0 | 48.6 |
| Explain the transformer architecture in simple terms. | 10.4 | 444.1 | 49.3 |
| How do recent papers approach learning rate scheduling? | 10.1 | 475.5 | 49.4 |
| What are the key differences between RNNs and Transformers? | 10.0 | 464.6 | 50.1 |
| Summarize the attention mechanism in one paragraph. | 12.6 | 481.5 | 50.8 |

---

## Reproducibility

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model
ollama pull phi4-mini

# 3. Run benchmark (from repo root, .venv activated)
make benchmark
# or directly:
python scripts/benchmark_ollama.py --model phi4-mini --runs 3
```

Results are saved to `docs/benchmarks.json` and the summary table printed to stdout.
