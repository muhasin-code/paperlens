.PHONY: install install-dev lint format test run clean


# ─── Setup ────────────────────────────────────────────────────────────────────────

install:
	@echo "-> Installing runtime dependencies (CPU torch first)..."
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

install-dev: install
	@echo "-> Installing development tools..."
	pip install -r requirements-dev.txt
	pre-commit install
	@echo " Dev environment ready. Run 'make lint' to verify."


# ─── Code Quality ─────────────────────────────────────────────────────────────────

lint:
	@echo "-> Running ruff linter..."
	ruff check src/ tests/

format:
	@echo "-> Running ruff formatter..."
	ruff format src/ tests/
	ruff check --fix src/ tests/


# ─── Testing ──────────────────────────────────────────────────────────────────────

test:
	@echo "-> Running pytest..."
	pytest tests/ -v


# ─── Ingestion ────────────────────────────────────────────────────────

ingest:
	@echo "-> Running full arXiv ingestion (~1-2 hours for 500 papers) ..."
	python scripts/ingest.py

ingest-dry:
	@echo "-> Dry run: fetching metadata only, no downloads ..."
	python scripts/ingest.py --dry-run


# ─── Parsing ──────────────────────────────────────────────────────

parse:
	@echo "-> Parsing PDFs into section-aware chunks (~30-60 min for 500 papers)..."
	python scripts/parse.py

parse-dry:
	@echo "-> Dry run: parsing first 5 papers, no writes..."
	python scripts/parse.py --dry-run --limit 5


# ─── Embedding ───────────────────────────────────────────────────────────

embed:
	@echo "-> Embedding chunks into ChromaDB (~10-20 minutes for 17k chunks)..."
	python scripts/embed.py

embed-dry:
	@echo "-> Dry run: embedding first 5 chunks, no writes..."
	python scripts/embed.py --dry-run --limit 5


# ─── BM25 Keyword Index (Phase 2) ───────────────────────────────────────────────────────────

bm25-build:
	@echo "-> Building BM25 keyword index..."
	python scripts/build_bm25.py;

bm25-rebuild:
	@echo "-> Rebuilding BM25 keyword index..."
	python scripts/build_bm25.py --rebuild

bm25-clean:
	@echo "-> Removing BM25 index..."
	rm -f data/bm25_index.pkl
	@echo "BM25 index removed."


# ─── Hybrid Retrieval (Phase 2.2) ─────────────────────────────────────────────

hybrid-verify:
	@echo "-> Verifying hybrid retrieval..."
	python scripts/verify_hybrid.py


# ─── Cross-Encoder Reranking (Phase 2.3) ─────────────────────────────────────────────

reranker-verify:
	@echo "-> Verifying cross-encoder reranking..."
	python scripts/verify_reranker.py


# ─── Citation Enforcement (Phase 2.4) ─────────────────────────────────────────────

prompt-verify:
	@echo "-> Verifying prompt loading..."
	python scripts/verify_prompt.py


# ─── Retrieval Quality Benchmark (Phase 2.5) ─────────────────────────────────────────────

eval-retrieval:
	@echo "-> Running retrieval quality benchmark..."
	@echo "   This runs in HEADLESS MODE (no Ollama/LLM required)."
	python scripts/evaluate_retrieval.py


# ─── Development Server ───────────────────────────────────────────────────────────

run-backend:
	@echo "-> Starting FastAPI dev server on http://localhost:8000 ..."
	uvicorn src.paperlens.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	@echo "-> Starting the React backend server on http://localhost:3000 ..."
	cd frontend/ && npm run dev


# ─── Benchmarking ─────────────────────────────────────────────────────────────────

benchmark:
	@echo "-> Running Ollama inference benchmarks (phi4-mini + llama3.2:1b)..."
	@echo "   Ensure 'ollama serve' is running and models are pulled."
	python scripts/benchmark_ollama.py


# ─── Cleanup ──────────────────────────────────────────────────────────────────────

clean:
	@echo "-> Cleaning build artifacts and caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null ||true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/
	@echo "Clean"


# ─── Help ─────────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo " make install			Install runtime dependencies (CPU torch + requirements.txt)"
	@echo " make install-dev 		Install runtime + dev tools + pre-commit hooks"
	@echo " make lint 			Ruff lint check (no changes)"
	@echo " make format			Ruff format + auto-fix (modifies files)"
	@echo " make test			Run pytest suite"
	@echo " make bm25-build     Build BM25 keyword index from chunks.parquet"
	@echo " make bm25-rebuild	Rebuild BM25 keyword index from chunks.parquet forcefully"
	@echo " make bm25-clean     Remove data/bm25_index.pkl"
	@echo " make hybrid-verify    Smoke-test: hybrid retrieval with RRF"
	@echo " make reranker-verify    Smoke-test: cross-encoder reranking"
	@echo " make prompt-verify      Smoke-test: prompt loader"
	@echo " make eval-retrieval      Run full retrieval benchmark (baseline vs hybrid vs hybrid+reranker)"
	@echo " make run-backend		Start FastAPI dev server"
	@echo " make run-frontend		Start React dev server"
	@echo " make benchmark      		Run Ollama inference benchmarks (tokens/sec, TTFT, RAM)"
	@echo " make clean			Remove __pycache__, .pyc, pytest and coverage artifacts"
	@echo " make ingest			Run full arXiv ingestion pipeline"
	@echo " make ingest-dry			Preview what would be ingested (no downloads)"
	@echo " make parse			Run full PDF parsing pipeline"
	@echo " make parse-dry			Preview parsing on 5 papers (no writes)"
	@echo " make embed			Run full embedding pipeline into ChromaDB"
	@echo " make embed-dry 		Preview embedding on 5 chunks (no writes)"
