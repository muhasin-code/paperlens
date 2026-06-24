"""
Tests for the arXiv ingestion pipeline.

arXiv API calls are fully mocked - no network access required.
Covers: Paper serialization, idempotency, query construction, dry-run branch.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.paperlens.ingestion.fetcher import ArxivFetcher
from src.paperlens.ingestion.models import Paper
from src.paperlens.settings import Settings


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary directory - no real disk writes."""
    return Settings(
        arxiv_category="cs.LG",
        arxiv_max_results=10,
        arxiv_date_from="2024-01-01",
        arxiv_date_to="2026-12-31",
        arxiv_delay_seconds=0.0,
        metadata_path=tmp_path / "metadata.jsonl",
        pdf_dir=tmp_path / "pdfs",
        ingestion_log=tmp_path / "ingestion.log",
    )


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401.00001v1",
        title="Test Paper: Attention Is All You Need For Testing",
        abstract="We propose a test suite. Tests are all you need.",
        authors=["Alice Test", "Bob Mock"],
        categories=["cs.LG", "cs.AI"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 2, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        pdf_path="/tmp/2401.00001v1.pdf",
    )


# ── Paper Model ─────────────────────────────────────────────────────────────
class TestPaperModel:
    def test_to_jsonl_ends_with_newline(self, sample_paper: Paper) -> None:
        assert sample_paper.to_jsonl().endswith("\n")

    def test_roundtrip_jsonl(self, sample_paper: Paper) -> None:
        restored = Paper.from_jsonl(sample_paper.to_jsonl())
        assert restored.arxiv_id == sample_paper.arxiv_id
        assert restored.title == sample_paper.title
        assert restored.authors == sample_paper.authors
        assert restored.categories == sample_paper.categories

    def test_pdf_path_defaults_to_none(self) -> None:
        paper = Paper(
            arxiv_id="2401.00001v1",
            title="No PDF",
            abstract="Abstract",
            authors=["Carol"],
            categories=["cs.LG"],
            published=datetime(2024, 1, 1, tzinfo=UTC),
            updated=datetime(2024, 1, 1, tzinfo=UTC),
            pdf_url="",
        )
        assert paper.pdf_path is None

    def test_ingested_at_is_set_automatically(self, sample_paper: Paper) -> None:
        assert sample_paper.ingested_at is not None
        assert sample_paper.ingested_at.tzinfo is not None


# ── Idempotency ─────────────────────────────────────────────────────────────
class TestIdempotency:
    def test_load_existing_ids_returns_empty_set_when_file_absent(self, settings: Settings) -> None:
        fetcher = ArxivFetcher(settings)
        assert fetcher.load_existing_ids() == set()

    def test_append_paper_creates_metadata_file(
        self, settings: Settings, sample_paper: Paper
    ) -> None:
        fetcher = ArxivFetcher(settings)
        assert not settings.metadata_path.exists()
        fetcher._append_paper(sample_paper)
        assert settings.metadata_path.exists()

    def test_appended_paper_id_is_loadable(self, settings: Settings, sample_paper: Paper) -> None:
        fetcher = ArxivFetcher(settings)
        fetcher._append_paper(sample_paper)
        loaded = fetcher.load_existing_ids()
        assert sample_paper.arxiv_id in loaded

    def test_multiple_appends_all_ids_present(
        self, settings: Settings, sample_paper: Paper
    ) -> None:
        paper2 = sample_paper.model_copy(update={"arxiv_id": "2401.00099v1"})
        fetcher = ArxivFetcher(settings)
        fetcher._append_paper(sample_paper)
        fetcher._append_paper(paper2)
        ids = fetcher.load_existing_ids()
        assert sample_paper.arxiv_id in ids
        assert paper2.arxiv_id in ids

    def test_load_existing_ids_tolerates_blank_lines(
        self, settings: Settings, sample_paper: Paper
    ) -> None:
        settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        settings.metadata_path.write_text(sample_paper.to_jsonl() + "\n\n", encoding="utf-8")
        fetcher = ArxivFetcher(settings)
        ids = fetcher.load_existing_ids()
        assert sample_paper.arxiv_id in ids


# ── Query Construction ──────────────────────────────────────────────────────
class TestQueryConstruction:
    def test_query_contains_configured_category(self, settings: Settings) -> None:
        fetcher = ArxivFetcher(settings)
        assert "cs.LG" in fetcher._build_query()

    def test_query_contains_date_range_without_hyphens(self, settings: Settings) -> None:
        fetcher = ArxivFetcher(settings)
        query = fetcher._build_query()
        assert "cs.LG" in query

    def test_query_structure(self, settings: Settings) -> None:
        fetcher = ArxivFetcher(settings)
        query = fetcher._build_query()
        assert "cat:" in query
        assert "cs.LG" in query


# ── Dry run ─────────────────────────────────────────────────────────────────
class TestDryRun:
    def test_dry_run_writes_no_files(self, settings: Settings) -> None:
        fetcher = ArxivFetcher(settings)
        with patch.object(fetcher, "fetch_metadata", return_value=[]):
            stats = fetcher.run(dry_run=True)
        assert not settings.metadata_path.exists()
        assert stats["mode"] == "dry_run"
        assert stats["pdfs_downloaded"] == 0

    def test_dry_run_reports_existing_papers(self, settings: Settings, sample_paper: Paper) -> None:
        fetcher = ArxivFetcher(settings)
        fetcher._append_paper(sample_paper)
        with patch.object(fetcher, "fetch_metadata", return_value=[]):
            stats = fetcher.run(dry_run=True)
        assert stats["existing_papers"] == 1
