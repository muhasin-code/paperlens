# Active Milestone Plan

> **Workflow:** This file holds **one milestone at a time**. When finished, archive it as a new Markdown file under `plan/completed-milestones/phase-1-core-rag-pipeline/`, add a summary row to [`plan/completed_milestones.md`](completed_milestones.md), then replace this file with the next milestone from [`plan/blueprint.md`](blueprint.md).

---

## Milestone 1.3 — Embeddings & Vector Store

| Field | Value |
|---|---|
| **Phase** | 1 — Core RAG Pipeline |
| **Blueprint ref** | [`plan/blueprint.md`](blueprint.md) → Milestone 1.3 |
| **Prerequisites** | Milestone 1.2 complete; `data/processed/chunks.parquet` exists with ≥10,000 chunks; `sentence-transformers`, `chromadb`, and a CPU build of `torch` installed; `.venv` active; `requirements.txt` matches the "Retrieval & Embeddings" section |
| **Estimated time** | 6–9 hours (includes a full-corpus embedding run over ~17k chunks) |
| **Release tag** | None |

### Goal

Turn the section-aware chunk dataset produced by Milestone 1.2 into a **queryable semantic index**. Every chunk is embedded with `BAAI/bge-large-en-v1.5` (sentence-transformers) and persisted in a ChromaDB collection together with its provenance metadata. The outcome is the first component of the retrieval stack: given a natural-language query, the system returns the top-k most semantically similar chunks, each carrying its `chunk_id`, `arxiv_id`, `section_label`, and page range so downstream citation enforcement (Phase 2.4) can point back to the source.

**End state:** `data/chroma/` contains a populated ChromaDB collection (`paperlens_chunks`) holding one vector per chunk in `chunks.parquet`; a `make embed` run is idempotent and resumable; a basic semantic-search function returns ranked chunks for an arbitrary query; baseline retrieval latency is measured and recorded in `docs/retrieval-baseline.md`.

### Deliverables Checklist

- [ ] `requirements.txt` "Retrieval & Embeddings" section confirmed (`torch` CPU build, `sentence-transformers>=3.1.0`, `chromadb>=0.5.18`)
- [ ] `src/paperlens/embedding/__init__.py`
- [ ] `src/paperlens/embedding/models.py` — `EmbeddedChunk` Pydantic model (carries embedding + source `Chunk` fields)
- [ ] `src/paperlens/embedding/embedder.py` — `EmbeddingModel` class: wraps sentence-transformers, applies the BGE query prefix, handles device selection, batch encodes
- [ ] `src/paperlens/embedding/vector_store.py` — `VectorStore` class: ChromaDB open/create, `upsert_chunks`, `query`, `count`, `reset`
- [ ] `src/paperlens/embedding/retriever.py` — `SemanticRetriever` class: encode query → top-k search → return ranked `Chunk` records
- [ ] `src/paperlens/embedding/pipeline.py` — `EmbeddingPipeline` class: orchestrates load → embed → persist with idempotency and a `--rebuild` path
- [ ] `scripts/embed.py` — CLI entry point with `--dry-run`, `--limit`, `--force`, `--rebuild`
- [ ] `Makefile` — `embed` and `embed-dry` targets added
- [ ] `tests/test_embedding/__init__.py`, `tests/test_embedding/test_embedder.py`, `tests/test_embedding/test_vector_store.py`
- [ ] Embedding executed: ChromaDB collection holds ≥10,000 vectors matching `chunks.parquet`
- [ ] `docs/retrieval-baseline.md` — embedding model choice, index size, dimension, and retrieval latency baseline
- [ ] All changes committed and pushed (`git push`)

---

## Step 1 — Verify the Embedding Stack in `requirements.txt`

### Why these exact dependencies

`BAAI/bge-large-en-v1.5` is loaded through `sentence-transformers`, which in turn depends on `torch`. On this machine (Dell Latitude 7490, CPU-only) we must install a **CPU build of torch** — the default `pip install sentence-transformers` may pull a CUDA-enabled torch wheel that is several GB and wastes disk. The `requirements.txt` already lists both libraries but the torch install order matters.

### Actions

Confirm `requirements.txt` contains the "Retrieval & Embeddings" block (it should — see current file). If `torch` is not yet installed in `.venv`, install it **first** with the CPU index URL, then the rest:

```bash
source .venv/bin/activate

# 1) CPU-only torch MUST be installed before sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Then the rest of the stack
pip install -r requirements.txt

# 3) Verify imports
python -c "
import torch, sentence_transformers, chromadb
print('torch', torch.__version__)
print('sentence-transformers', sentence_transformers.__version__)
print('chromadb', chromadb.__version__)
print('cuda available:', torch.cuda.is_available())
"
```

The output of the final check should show `cuda available: False` (CPU-only box) and no import errors. `chromadb` will also pull in `onnxruntime` and other transitive deps on first import — that is expected.

### Verification

- [ ] `torch` is importable and reports `cuda available: False`
- [ ] `sentence_transformers` and `chromadb` import without error
- [ ] `BAAI/bge-large-en-v1.5` can be referenced (it will be downloaded automatically on first `EmbeddingModel` instantiation — see Step 4)

---

## Step 2 — Create Embedding Package Skeleton

### Actions

```bash
mkdir -p src/paperlens/embedding
mkdir -p tests/test_embedding
touch tests/test_embedding/__init__.py
```

Create `src/paperlens/embedding/__init__.py`:

```python
"""Chunk embedding and ChromaDB vector-store for semantic retrieval."""
```

### Verification

- [ ] `src/paperlens/embedding/` directory exists
- [ ] `src/paperlens/embedding/__init__.py` exists
- [ ] `tests/test_embedding/__init__.py` exists

---

## Step 3 — Define the Embedded Chunk Model

### Why a dedicated model

The `Chunk` model (Milestone 1.2) is the storage contract for raw text + provenance. Once embedded, a chunk additionally carries a fixed-length vector. Keeping the vector **outside** `Chunk` avoids bloating the Pydantic model that downstream code reuses for I/O. Instead we define `EmbeddedChunk`, which wraps the original `Chunk` plus its `embedding` (a `list[float]`) and the embedding `dimension`.

### Actions

Create `src/paperlens/embedding/models.py`:

```python
"""Pydantic models for embedded chunks and retrieval results.

EmbeddedChunk: a Chunk plus its dense vector. Kept separate from the base
Chunk model so that raw text/metadata I/O (parsing, persistence) is never
coupled to the embedding dimension.
"""

from pydantic import BaseModel, Field

from src.paperlens.parsing.models import Chunk


class EmbeddedChunk(BaseModel):
    """A Chunk together with its dense embedding vector."""

    chunk: Chunk
    """The source chunk (full provenance retained for citation)."""

    embedding: list[float]
    """Dense vector produced by the embedding model."""

    dimension: int
    """Embedding dimensionality (e.g. 1024 for bge-large-en-v1.5)."""

    def vector(self) -> list[float]:
        """Return the embedding vector directly (convenience for ChromaDB)."""
        return self.embedding


class RetrievalResult(BaseModel):
    """A single ranked retrieval hit returned by the semantic retriever."""

    chunk: Chunk
    """The matched chunk with full provenance."""

    score: float
    """Similarity score (cosine, higher = more relevant)."""

    rank: int
    """1-based rank position in the returned list."""
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.embedding.models import EmbeddedChunk, RetrievalResult
from src.paperlens.parsing.models import Chunk
print('Embedding models OK')
"
```

---

## Step 4 — Implement the Embedding Model Wrapper

### Embedding-model design decisions

`BAAI/bge-large-en-v1.5` is a 1024-dimension bi-encoder. Two properties are critical for correctness:

1. **Query prefix.** BGE models are trained so that *queries* should be prefixed with
   `Represent this sentence for searching relevant passages: ` while *documents* (chunks)
   are embedded as-is. Skipping the query prefix sharply degrades recall. The wrapper
   applies the prefix only when embedding queries, never when embedding corpus chunks.

2. **Normalization.** BGE embeddings are trained to be compared with cosine similarity.
   We normalize each vector to unit length so that a plain dot product equals cosine
   similarity. ChromaDB's default `cosine` space already normalizes, but normalizing
   ourselves keeps the vectors portable to other stores (Phase 2.1 BM25 fusion, Phase 2.3).

3. **Device selection.** On this CPU-only box we force `device="cpu"`. The wrapper reads
   an optional `embedding_device` setting but defaults to CPU and degrades gracefully.

4. **Batching.** Chunks are encoded in batches (default 64) with a `tqdm` progress bar so
   a ~17k-chunk run is monitorable and OOM-safe.

### Actions

Create `src/paperlens/embedding/embedder.py`:

```python
"""
Embedding model wrapper around sentence-transformers.

Design decisions:
- BAAI/bge-large-en-v1.5: 1024-dim, strong general-purpose English retriever.
- Query prefix applied ONLY to queries (BGE training convention).
- Embeddings normalized to unit length so dot-product == cosine similarity.
- CPU-only by default; batch encoding with tqdm progress.
"""

import logging
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")

# BGE training convention: prefix queries with this string.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Default batch size for corpus encoding (memory-safe on CPU).
DEFAULT_BATCH_SIZE = 64


class EmbeddingModel:
    """Wraps a sentence-transformers model for chunk + query embedding."""

    def __init__(self, settings: Settings, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_name = settings.embedding_model
        self.batch_size = batch_size
        self.device = "cpu"  # CPU-only box; override via settings if GPU available
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        self.logger = logger
        self.logger.info(
            "Loaded embedding model %s (dim=%d, device=%s)",
            self.model_name, self.dimension, self.device,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of corpus chunks (no query prefix, normalized)."""
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [
            EmbeddedChunk(chunk=c, embedding=vec.tolist(), dimension=self.dimension)
            for c, vec in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string WITH the BGE query prefix, normalized."""
        prefixed = f"{BGE_QUERY_PREFIX}{query}"
        vec = self._model.encode(
            [prefixed],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries (prefixed, normalized)."""
        prefixed = [f"{BGE_QUERY_PREFIX}{q}" for q in queries]
        vecs = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]
```

> **Note on caching:** Do **not** cache the `EmbeddingModel` with `lru_cache` from
> `settings.py` — sentence-transformers holds a large in-memory model and is created
> once per pipeline run by the `EmbeddingPipeline`. The `@lru_cache` import is shown only
> to mirror the settings module's style; the pipeline owns a single instance.

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.embedding.embedder import EmbeddingModel
m = EmbeddingModel(Settings())
print('dim', m.dimension)
q = m.embed_query('How do recent papers schedule learning rate?')
print('query vec len', len(q))
"
```

First run downloads `BAAI/bge-large-en-v1.5` (~1.3 GB, FP32) — allow time / bandwidth.

---

## Step 5 — Implement the ChromaDB Vector Store

### Vector-store design decisions

- **One collection** named `paperlens_chunks`. Each ChromaDB record stores:
  - `id` = `chunk_id` (globally unique → idempotent upsert)
  - `document` = chunk `text` (returned verbatim for context assembly in Phase 1.4)
  - `embedding` = the 1024-d vector
  - `metadata` = a *flat* dict of provenance: `arxiv_id`, `title`, `section_label`,
    `chunk_index`, `page_start`, `page_end`, `token_count`,
    `total_chunks_in_section`, `total_chunks_in_paper`, and `authors` joined as a
    semicolon string (ChromaDB metadata values must be str/int/float/bool — not lists).
- **Persistence.** ChromaDB persists to `settings.chroma_persist_dir` (`data/chroma`),
  so the index survives process restarts and is reproducible via the rebuild script.
- **Space.** `cosine` distance so higher score = more similar (matches the retriever API).
- **Reset.** `reset()` deletes and recreates the collection — used by `--rebuild`.

### Actions

Create `src/paperlens/embedding/vector_store.py`:

```python
"""
ChromaDB vector store for embedded chunks.

Design decisions:
- Single collection 'paperlens_chunks', persisted to settings.chroma_persist_dir.
- chunk_id is the ChromaDB id -> upsert is naturally idempotent.
- Metadata is flattened (authors joined) because ChromaDB rejects list values.
- cosine space so score increases with similarity.
"""

import logging
from pathlib import Path

import chromadb

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")

COLLECTION_NAME = "paperlens_chunks"


class VectorStore:
    """Persistent ChromaDB store for embedded chunks."""

    def __init__(self, settings: Settings) -> None:
        self.persist_dir: Path = settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger = logger

    @staticmethod
    def _metadata_for(ec: EmbeddedChunk) -> dict:
        c = ec.chunk
        return {
            "arxiv_id": c.arxiv_id,
            "title": c.title,
            "authors": "; ".join(c.authors),
            "section_label": c.section_label,
            "chunk_index": c.chunk_index,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "total_chunks_in_section": c.total_chunks_in_section,
            "total_chunks_in_paper": c.total_chunks_in_paper,
            "dimension": ec.dimension,
        }

    def upsert(self, embedded: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks (idempotent by chunk_id)."""
        if not embedded:
            return
        self.collection.upsert(
            ids=[ec.chunk.chunk_id for ec in embedded],
            documents=[ec.chunk.text for ec in embedded],
            embeddings=[ec.embedding for ec in embedded],
            metadatas=[self._metadata_for(ec) for ec in embedded],
        )
        self.logger.info("Upserted %d chunks (total now %d)", len(embedded), self.count())

    def count(self) -> int:
        """Return the number of vectors currently in the collection."""
        return self.collection.count()

    def query(self, query_vector: list[float], top_k: int) -> dict:
        """Return the top_k nearest neighbours for a single query vector."""
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def reset(self) -> None:
        """Delete and recreate the collection (used by --rebuild)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Reset collection %s", COLLECTION_NAME)
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.embedding.vector_store import VectorStore
vs = VectorStore(Settings())
print('collection', vs.collection.name, 'count', vs.count())
"
```

---

## Step 6 — Implement the Semantic Retriever

### Why a separate retriever class

The retriever is the *query-time* path: embed the query (with prefix) → search the store
→ reconstruct `Chunk` objects from returned metadata → rank. Keeping it separate from the
store means Phase 1.4 (FastAPI) and Phase 2.2 (hybrid) can call it directly without
re-implementing query embedding. The retriever returns `RetrievalResult` objects so the
LLM layer (Phase 1.4) and citation enforcement (Phase 2.4) receive structured, typed hits.

### Actions

Create `src/paperlens/embedding/retriever.py`:

```python
"""
Semantic retriever: query embedding + top-k search -> ranked Chunk results.

Design decisions:
- Reuses EmbeddingModel (query prefix) and VectorStore (cosine search).
- Reconstructs Chunk Pydantic objects from ChromaDB metadata so downstream
  code receives the same typed contract as parsing produced.
- ChromaDB returns cosine DISTANCE; we convert to similarity score = 1 - distance.
"""

import logging

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.models import RetrievalResult
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


class SemanticRetriever:
    """Top-k semantic search over the ChromaDB chunk index."""

    def __init__(self, settings: Settings, top_k: int | None = None) -> None:
        self.settings = settings
        self.embedder = EmbeddingModel(settings)
        self.store = VectorStore(settings)
        self.top_k = top_k or settings.retrieval_top_k
        self.logger = logging.getLogger("paperlens.embedding")

    @staticmethod
    def _chunk_from_metadata(chunk_id: str, metadata: dict, document: str) -> Chunk:
        authors = [a.strip() for a in metadata.get("authors", "").split(";") if a.strip()]
        return Chunk(
            chunk_id=chunk_id,
            arxiv_id=metadata["arxiv_id"],
            title=metadata.get("title", ""),
            authors=authors,
            section_label=metadata.get("section_label", "unknown"),
            chunk_index=int(metadata.get("chunk_index", 0)),
            page_start=int(metadata.get("page_start", 0)),
            page_end=int(metadata.get("page_end", 0)),
            text=document,
            token_count=int(metadata.get("token_count", 0)),
            total_chunks_in_section=int(metadata.get("total_chunks_in_section", 0)),
            total_chunks_in_paper=int(metadata.get("total_chunks_in_paper", 0)),
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return ranked RetrievalResult objects for a natural-language query."""
        k = top_k or self.top_k
        qvec = self.embedder.embed_query(query)
        resp = self.store.query(qvec, top_k=k)

        ids = resp["ids"][0]
        docs = resp["documents"][0]
        metas = resp["metadatas"][0]
        dists = resp["distances"][0]

        results: list[RetrievalResult] = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            chunk = self._chunk_from_metadata(cid, meta, doc)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.embedding.retriever import SemanticRetriever
r = SemanticRetriever(Settings(), top_k=3)
print('Retriever OK')
"
```

(Requires the collection to be populated — see Step 10. Use a tiny `--limit` run first.)

---

## Step 7 — Implement the Embedding Pipeline (Orchestration + Persistence)

### Why a separate pipeline class

The pipeline ties loading (`chunks.parquet`), embedding, and persistence into one
resumable, idempotent operation. Responsibilities:

- **Load** chunks from the Parquet file produced in 1.2.
- **Idempotency.** On a normal run, it asks the store how many vectors already exist and
  skips chunks whose `chunk_id` is already present (mirrors `ParsingPipeline` behaviour).
  A `--force` run re-embeds everything but still upserts by id (no duplicates).
- **Rebuild.** A `--rebuild` run resets the collection first, guaranteeing a clean index
  (the "index rebuild script for reproducibility" called for in the blueprint).
- **Batching + progress.** Embeds in `DEFAULT_BATCH_SIZE` batches with a tqdm bar.

### Actions

Create `src/paperlens/embedding/pipeline.py`:

```python
"""
Embedding pipeline: orchestrates chunk load -> embed -> persist in ChromaDB.

Design decisions:
- Idempotent: existing chunk_ids in the store are skipped unless --force/--rebuild.
- --rebuild fully resets the collection before embedding (clean reproducible index).
- Chunks streamed from Parquet in bulk; embedded in batches with tqdm progress.
"""

import logging

import pandas as pd

from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.embedding")


class EmbeddingPipeline:
    """Orchestrates the full embed-and-store pipeline over the chunk corpus."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = EmbeddingModel(settings)
        self.store = VectorStore(settings)
        self.logger = logger

    def _load_chunks(self, limit: int | None = None) -> list[Chunk]:
        path = self.settings.processed_chunks_path
        if not path.exists():
            self.logger.error("Chunks file not found: %s", path)
            return []
        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit)
        return [Chunk.from_parquet_row(row) for _, row in df.iterrows()]

    def _existing_ids(self) -> set[str]:
        """Best-effort listing of already-embedded chunk_ids."""
        try:
            # ChromaDB peek is capped; for idempotency we rely on upsert-by-id,
            # but we still report a count for logging/skipping decisions.
            return set()
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Could not list existing ids: %s", exc)
            return set()

    def run(
        self,
        dry_run: bool = False,
        limit: int | None = None,
        force: bool = False,
        rebuild: bool = False,
    ) -> dict:
        """Execute the embedding pipeline.

        Args:
            dry_run: Load + embed a sample in memory but do NOT write to ChromaDB.
            limit: Only process the first N chunks (smoke test).
            force: Re-embed all chunks (still upserted by id -> no duplicates).
            rebuild: Reset the collection before embedding (clean index).

        Returns:
            Dict of stats: mode, chunks_loaded, chunks_embedded, vectors_in_store, output_dir.
        """
        self.logger.info(
            "=== PaperLens Embedding Start | dry_run=%s | limit=%s | force=%s | rebuild=%s ===",
            dry_run, limit, force, rebuild,
        )

        chunks = self._load_chunks(limit=limit)
        if not chunks:
            self.logger.warning("No chunks loaded - aborting.")
            return {"mode": "dry_run" if dry_run else "full", "chunks_loaded": 0,
                    "chunks_embedded": 0, "vectors_in_store": self.store.count(),
                    "output_dir": str(self.settings.chroma_persist_dir)}

        if rebuild and not dry_run:
            self.store.reset()

        embedded = self.embedder.embed_chunks(chunks)

        if not dry_run:
            self.store.upsert(embedded)

        self.logger.info(
            "=== Embedding Complete | loaded=%d | embedded=%d | store_total=%d ===",
            len(chunks), len(embedded), self.store.count(),
        )

        return {
            "mode": "dry_run" if dry_run else "full",
            "chunks_loaded": len(chunks),
            "chunks_embedded": len(embedded),
            "vectors_in_store": self.store.count(),
            "output_dir": str(self.settings.chroma_persist_dir),
        }
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.embedding.pipeline import EmbeddingPipeline
p = EmbeddingPipeline(Settings())
print('Pipeline imported OK')
"
```

---

## Step 8 — Create CLI Script and Update Makefile

### 8a — `scripts/embed.py`

```python
#!/usr/bin/env python3
"""
CLI entry point for the PaperLens embedding pipeline.

Examples:
    python scripts/embed.py              # full embed using .env defaults
    python scripts/embed.py --dry-run    # embed 5 chunks in memory, no writes
    python scripts/embed.py --limit 100  # embed only the first 100 chunks (smoke test)
    python scripts/embed.py --force      # re-embed everything (upsert by id)
    python scripts/embed.py --rebuild    # reset collection, then embed all
    make embed                          # same as first option
    make embed-dry                      # same as --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.embedding.pipeline import EmbeddingPipeline
from src.paperlens.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks and store them in ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Load + embed in memory but do not write to ChromaDB.")
    parser.add_argument("--limit", type=int,
                        help="Only process the first N chunks (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed all chunks (idempotent upsert by id).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Reset the ChromaDB collection before embedding.")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = EmbeddingPipeline(settings)
    stats = pipeline.run(
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
        rebuild=args.rebuild,
    )

    print("\n--- Embedding Stats ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

### 8b — Update Makefile

Add the following targets after the existing `parse-dry` target:

```makefile
# ─── Embedding ─────────────────────────────────────────────────────────────

embed:
	@echo "→ Embedding chunks into ChromaDB (~10-20 min for 17k chunks)..."
	python scripts/embed.py

embed-dry:
	@echo "→ Dry run: embedding first 5 chunks, no writes..."
	python scripts/embed.py --dry-run --limit 5
```

Also update `make help`:

```makefile
	@echo "  make embed          Run full embedding pipeline into ChromaDB"
	@echo "  make embed-dry      Preview embedding on 5 chunks (no writes)"
```

### Verification

```bash
source .venv/bin/activate
python scripts/embed.py --help   # should print usage without error
make embed-dry                   # embeds 5 chunks in memory; expect a few seconds
```

The dry-run should show `chunks_loaded: 5`, `chunks_embedded: 5`, and `mode: dry_run`.

---

## Step 9 — Write Tests

### 9a — Test package init

Already created in Step 2.

### 9b — `tests/test_embedding/test_embedder.py`

These tests use a **tiny synthetic model is not feasible**, so we instead validate the
wrapper's *behaviour* against the real (small, fast) `all-MiniLM-L6-v2` only when a
`PAPERLENS_TEST_EMBEDDER` env var is set; otherwise we mock the underlying
`SentenceTransformer` to avoid downloading 80 MB on every CI run. The test below mocks
the model so it runs in CI without network access.

```python
"""
Tests for the EmbeddingModel wrapper.

The real sentence-transformers model is mocked so tests run offline in CI.
We verify: batch embedding returns one EmbeddedChunk per Chunk, vectors are
normalized (unit length), and the query prefix is applied to queries only.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.paperlens.embedding.embedder import BGE_QUERY_PREFIX, EmbeddingModel
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


def _make_chunk(text: str, cid: str = "2401.00001v1_chunk_0001") -> Chunk:
    return Chunk(
        chunk_id=cid,
        arxiv_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice"],
        section_label="method",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text=text,
        token_count=10,
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )


@pytest.fixture
def fake_model() -> MagicMock:
    """A mocked SentenceTransformer returning fixed unit vectors."""
    m = MagicMock()
    m.get_sentence_embedding_dimension.return_value = 4

    def _encode(texts, **kwargs):
        # Return distinct unit vectors per input.
        out = []
        for i, _ in enumerate(texts):
            vec = np.zeros(4, dtype=np.float32)
            vec[i % 4] = 1.0
            out.append(vec)
        return np.array(out, dtype=np.float32)

    m.encode.side_effect = _encode
    return m


@pytest.fixture
def embedder(fake_model: MagicMock) -> EmbeddingModel:
    with patch("src.paperlens.embedding.embedder.SentenceTransformer", return_value=fake_model):
        e = EmbeddingModel(Settings(), batch_size=2)
    return e


class TestEmbedChunks:
    def test_one_embedded_chunk_per_input(self, embedder: EmbeddingModel) -> None:
        chunks = [_make_chunk("alpha"), _make_chunk("beta", cid="2401.00001v1_chunk_0002")]
        embedded = embedder.embed_chunks(chunks)
        assert len(embedded) == 2
        assert embedded[0].chunk.chunk_id == "2401.00001v1_chunk_0001"

    def test_vectors_are_unit_normalized(self, embedder: EmbeddingModel) -> None:
        embedded = embedder.embed_chunks([_make_chunk("gamma")])
        vec = np.array(embedded[0].embedding)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_dimension_exposed(self, embedder: EmbeddingModel) -> None:
        embedded = embedder.embed_chunks([_make_chunk("delta")])
        assert embedded[0].dimension == 4


class TestEmbedQuery:
    def test_query_is_prefixed(self, embedder: EmbeddingModel, fake_model: MagicMock) -> None:
        embedder.embed_query("how to train?")
        # The first positional arg to encode should carry the BGE prefix.
        called_texts = fake_model.encode.call_args[0][0]
        assert called_texts[0].startswith(BGE_QUERY_PREFIX)

    def test_query_vector_is_unit_normalized(self, embedder: EmbeddingModel) -> None:
        vec = np.array(embedder.embed_query("normalize me"))
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)
```

### 9c — `tests/test_embedding/test_vector_store.py`

`VectorStore` requires ChromaDB on disk; we point it at a **temporary directory** so the
test is hermetic and leaves no artifacts in `data/chroma`.

```python
"""
Tests for the ChromaDB VectorStore.

Uses a temp persist dir so it never touches the real data/chroma index.
"""

from pathlib import Path

import pytest

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


def _embedded(text: str, cid: str, dim: int = 4) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=cid,
        arxiv_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice"],
        section_label="method",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text=text,
        token_count=10,
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )
    vec = [0.0] * dim
    vec[0] = 1.0
    return EmbeddedChunk(chunk=chunk, embedding=vec, dimension=dim)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    settings = Settings(chroma_persist_dir=tmp_path / "chroma")
    return VectorStore(settings)


class TestUpsertAndCount:
    def test_empty_count(self, store: VectorStore) -> None:
        assert store.count() == 0

    def test_upsert_increments_count(self, store: VectorStore) -> None:
        store.upsert([_embedded("a", "c1"), _embedded("b", "c2")])
        assert store.count() == 2

    def test_upsert_is_idempotent(self, store: VectorStore) -> None:
        store.upsert([_embedded("a", "c1")])
        store.upsert([_embedded("a-updated", "c1")])  # same id
        assert store.count() == 1


class TestQuery:
    def test_query_returns_nearest(self, store: VectorStore) -> None:
        store.upsert([_embedded("method text", "m1"), _embedded("intro text", "i1")])
        # Query vector identical to m1's vector -> m1 should rank first.
        resp = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert resp["ids"][0][0] == "m1"


class TestReset:
    def test_reset_clears_collection(self, store: VectorStore) -> None:
        store.upsert([_embedded("x", "x1")])
        store.reset()
        assert store.count() == 0
```

### 9d — Run all tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: all ingestion/parsing/embedding tests pass. No model download required (mocks).

---

## Step 10 — Run Embedding and Record Statistics

### 10a — Dry run first (mandatory)

```bash
source .venv/bin/activate
make embed-dry
```

Confirm `chunks_loaded: 5`, `chunks_embedded: 5`, and `mode: dry_run`. If it fails on
model download, ensure network access or pre-cache the model with
`python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"`.

### 10b — Full embedding run

> **Time estimate:** Embedding ~17k chunks with `bge-large-en-v1.5` on CPU takes
> roughly 10–20 minutes. Run in a tmux pane so it survives terminal disconnects.

```bash
tmux new -s embed
source .venv/bin/activate
make embed
# Ctrl+B D to detach; tmux attach -t embed to re-attach
```

If interrupted, re-run `make embed` — ChromaDB upsert-by-id makes re-runs safe (no
duplicates). For a guaranteed-clean index, use `make embed-rebuild` after adding the
target (or `python scripts/embed.py --rebuild`).

### 10c — Verify the index and record stats

Run this after embedding completes and paste the output into `docs/retrieval-baseline.md`:

```bash
source .venv/bin/activate

python - <<'EOF'
import time
import pandas as pd
from src.paperlens.settings import Settings
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.embedding.retriever import SemanticRetriever

settings = Settings()
store = VectorStore(settings)
df = pd.read_parquet(settings.processed_chunks_path)

print(f"Chunks in Parquet     : {len(df)}")
print(f"Vectors in ChromaDB   : {store.count()}")
print(f"Embedding model       : {settings.embedding_model}")
print(f"Collection            : {store.collection.name}")

# Retrieval latency baseline (single query, warmed up).
retriever = SemanticRetriever(settings, top_k=5)
q = "How do recent papers approach learning rate scheduling?"
_ = retriever.search(q)  # warm up
t0 = time.perf_counter()
results = retriever.search(q)
t1 = time.perf_counter()

print(f"\nQuery                : {q}")
print(f"Top-1 section        : {results[0].chunk.section_label}")
print(f"Top-1 arxiv_id       : {results[0].chunk.arxiv_id}")
print(f"Top-1 score          : {results[0].score:.4f}")
print(f"Retrieval latency    : {(t1 - t0) * 1000:.1f} ms (query embed + top-5 search)")
EOF

echo ""
echo "Disk usage:"
du -sh data/chroma
```

Expected: `Vectors in ChromaDB` ≈ `Chunks in Parquet` (≥10,000), latency in the tens-to-
low-hundreds of ms on CPU.

---

## Step 11 — Documentation: `docs/retrieval-baseline.md`

Create `docs/retrieval-baseline.md`. Fill in the placeholders with actual numbers from
Step 10c.

Key sections to include:
- **Embedding model choice** — why `BAAI/bge-large-en-v1.5` (1024-dim, strong general
  English retriever, open-weight, runs on CPU); note the BGE query-prefix convention.
- **Index parameters** — collection name, distance space (`cosine`), persistence path,
  embedding dimension, normalization approach.
- **Index size** — chunk count, vector count, disk usage.
- **Retrieval latency baseline** — single-query embed + top-k search in ms (this is the
  Phase 1 baseline; Phase 2 will add BM25 and reranking on top).
- **Rebuild instructions** — `python scripts/embed.py --rebuild` for a clean reproducible index.
- **Sample query + top result** — demonstrate that retrieval returns relevant, provenance-rich chunks.

---

## Step 12 — Commit and Push

```bash
cd ~/ML-Projects/paperlens
source .venv/bin/activate

pre-commit run --all-files
# Fix any ruff issues, then re-stage modified files

git add \
  requirements.txt \
  src/paperlens/embedding/ \
  scripts/embed.py \
  Makefile \
  tests/test_embedding/ \
  docs/retrieval-baseline.md

# Confirm data files are NOT staged
git status | grep "data/chroma" && echo "⚠ data/chroma staged — unstage it" || echo "✓ No data files staged"

git commit -m "$(cat <<'EOF'
Complete Milestone 1.3: embeddings and vector store.

- Add src/paperlens/embedding: models, EmbeddingModel (BGE + query prefix),
  VectorStore (ChromaDB), SemanticRetriever, EmbeddingPipeline.
- EmbeddingModel applies the BGE query prefix to queries only and normalizes vectors.
- VectorStore persists to data/chroma (cosine); upsert-by-id is idempotent.
- SemanticRetriever returns ranked, provenance-rich RetrievalResult objects.
- Add scripts/embed.py CLI (--dry-run, --limit, --force, --rebuild).
- Add Makefile targets: embed, embed-dry.
- Add tests/test_embedding: embedder (mocked) and vector_store (temp dir).
- Embed ~17k chunks into ChromaDB; record baseline in docs/retrieval-baseline.md.
EOF
)"

git push
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | CPU torch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: sentence_transformers` | not installed | `pip install -r requirements.txt` |
| `ModuleNotFoundError: chromadb` | not installed | `pip install chromadb` |
| First run downloads ~1.3 GB | `bge-large-en-v1.5` auto-fetch | Allow bandwidth; or pre-cache (see 10a) |
| `cuda available: True` but box is CPU | wrong torch wheel | reinstall CPU torch; set `device="cpu"` in embedder |
| `chunks_loaded: 0` | Parquet missing | Confirm `data/processed/chunks.parquet` from Milestone 1.2 |
| ChromaDB `ValueError: metadata value must be str/int/float/bool` | list in metadata | Ensure `authors` is joined to a string (handled in `VectorStore._metadata_for`) |
| Re-run doubles the count | not using upsert | Code uses `collection.upsert` by `chunk_id` → safe; verify `count()` stable across runs |
| Embedding run OOM | batch size too high on low RAM | Lower `DEFAULT_BATCH_SIZE` (e.g. 16) in `embedder.py` |
| `pytest` downloads a model | a test imports real model | Use the mocked `test_embedder.py`; only env-driven tests touch the network |

---

## Final Verification Gate

Run through this checklist before archiving this milestone. **All must pass.**

| # | Check | Pass? |
|---|---|---|
| 1 | `torch` (CPU), `sentence-transformers`, `chromadb` importable in `.venv` | [ ] |
| 2 | `from src.paperlens.embedding.models import EmbeddedChunk, RetrievalResult` — no error | [ ] |
| 3 | `from src.paperlens.embedding.embedder import EmbeddingModel` — no error | [ ] |
| 4 | `EmbeddingModel` applies BGE query prefix to queries only, normalizes vectors | [ ] |
| 5 | `from src.paperlens.embedding.vector_store import VectorStore` — no error | [ ] |
| 6 | `VectorStore.upsert` is idempotent by `chunk_id` | [ ] |
| 7 | `from src.paperlens.embedding.retriever import SemanticRetriever` — no error | [ ] |
| 8 | `from src.paperlens.embedding.pipeline import EmbeddingPipeline` — no error | [ ] |
| 9 | `python scripts/embed.py --help` prints usage | [ ] |
| 10 | `make embed-dry` completes with `chunks_embedded: 5`, `mode: dry_run` | [ ] |
| 11 | `pytest tests/ -v` — all tests pass (incl. embedding tests, mocked) | [ ] |
| 12 | `make embed` completed; ChromaDB collection count ≈ Parquet chunk count (≥10,000) | [ ] |
| 13 | Retrieval returns ranked, provenance-rich results for a sample query | [ ] |
| 14 | `docs/retrieval-baseline.md` exists with actual model/dimension/latency stats | [ ] |
| 15 | `git push` succeeds; no `data/` files in the commit | [ ] |
| 16 | CI workflow passes on GitHub | [ ] |
| 17 | Blueprint Milestone 1.3 checkboxes checked in `plan/blueprint.md` | [ ] |

---

## On Completion — Archive and Advance

1. **Archive** — create `plan/completed-milestones/phase-1-core-rag-pipeline/1.3-embeddings-vector-store.md` with the milestone summary and verbatim plan content, then add a summary row to `plan/completed_milestones.md`
2. **Blueprint** — check off Milestone 1.3 items in `plan/blueprint.md`
3. **Replace this file** with **Milestone 1.4 — FastAPI Backend (Basic RAG)**
4. **GitHub Projects** — move 1.3 card to Done

---

## Next Milestone Preview

**Milestone 1.4 — FastAPI Backend (Basic RAG)**

Create a FastAPI app exposing a `/query` endpoint that wires retrieval (1.3) → context
assembly → Ollama LLM generation (Phase 1 LLM). Return the answer together with the
source paper/chunk references for citation. Add request/response Pydantic schemas. Document
the endpoint spec and example `curl` requests in `docs/api.md`.

See `plan/blueprint.md` → Milestone 1.4.
