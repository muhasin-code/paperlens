"""
Application-wide settings loaded from the .env file.

All modules import get_settings() rather than reading env vars directly.
The Settings instance is cached after the first call via lru_cache.

Usage:
    from src.paperlens.settings import get_settings()
    settings = get_sttings()
    print(settings.arxiv_category)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860

    # ── arXiv Ingestion (Phase 1) ───────────────────────────────────────────
    arxiv_category: str = "cs.LG"
    arxiv_max_results: int = 500
    arxiv_date_from: str = "2023-01-01"
    arxiv_date_to: str = "2024-12-31"
    arxiv_delay_seconds: float = 3.0
    metadata_path: Path = Field(default=Path("./data/raw/metadata.jsonl"))
    pdf_dir: Path = Field(default=Path("./data/raw/pdfs"))
    ingestion_log: Path = Field(default=Path("./data/raw/ingestion.log"))

    # ── Retrieval (Phase 1–2) ───────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    bm25_index_path: Path = Field(default=Path("./data/bm25_index.pkl"))
    processed_chunks_path: Path = Field(default=Path("./data/processed/chunks.parquet"))
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"

    # ── Prompts (Phase 2) ───────────────────────────────────────────────────
    prompts_dir: Path = Field(default=Path("./configs/prompts"))
    prompt_version: str = "v1"

    # ── Langfuse (Phase 4) ──────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ── Evaluation (Phase 4) ────────────────────────────────────────────────
    ragas_faithfulness_threshold: float = 0.75
    gold_dataset_path: Path = Field(default=Path("./eval/gold_dataset.jsonl"))


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance. Called once per process."""
    return Settings()
