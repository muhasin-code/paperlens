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
	pytest tests/ -r


# ─── Development Server ───────────────────────────────────────────────────────────

run:
	@echo "-> Starting FastAPI dev server on http://localhost:8000 ..."
	uvicorn src.paperlens.main:app --reload --host 0.0.0.0 --port 8000


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
	@echo " make run			Start FastAPI dev server"
	@echo " make clean			Remove __pycache__, .pyc, pytest and coverage artifacts"
