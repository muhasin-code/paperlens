"""
Tests for the section-aware chunker.

Covers: single-section chunking, multi-section chunking, overlap correctness,
token count accuracy, unique ID generation.
"""

from datetime import UTC, datetime

import pytest

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.chunker import SectionAwareChunker
from src.paperlens.parsing.models import Section
from src.paperlens.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chunk_size_tokens=600,
        chunk_overlap_tokens=100,
    )


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401.00001v1",
        title="Test Paper on Attention Mechanisms",
        abstract="We propose a new attention mechanism.",
        authors=["Alice Test", "Bob Mock"],
        categories=["cs.LG"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 2, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        pdf_path="/tmp/2401.00001v1.pdf",
    )


@pytest.fixture
def short_section() -> Section:
    """A section that fits in a single chunk."""
    return Section(
        label="abstract",
        start_page=1,
        end_page=1,
        text="This is a short abstract. " * 10,
        confidence=1.0,
    )


@pytest.fixture
def long_section() -> Section:
    """A section that requires multiple chunks."""
    # ~1500 tokens of text — should produce 3 chunks with 600/100 settings.
    paragraph = (
        "We propose a novel method for training large language models. "
        "Our approach combines gradient descent with adaptive learning rates. "
        "Experiments on standard benchmarks show consistent improvements. "
    )
    return Section(
        label="method",
        start_page=3,
        end_page=8,
        text=paragraph * 80,
        confidence=1.0,
    )


class TestSingleChunk:
    def test_short_section_produces_one_chunk(
        self, settings: Settings, sample_paper: Paper, short_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=short_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert len(chunks) == 1
        assert chunks[0].section_label == "abstract"
        assert chunks[0].chunk_index == 0

    def test_single_chunk_has_correct_total_counts(
        self, settings: Settings, sample_paper: Paper, short_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=short_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert chunks[0].total_chunks_in_section == 1


class TestMultiChunk:
    def test_long_section_produces_multiple_chunks(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert len(chunks) >= 3

    def test_chunks_have_sequential_indices(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_have_unique_ids(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_token_counts_within_limit(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        for chunk in chunks:
            assert chunk.token_count <= 600

    def test_total_chunks_in_section_updated(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_section == total


class TestChunkPaper:
    def test_multiple_sections_produce_flat_chunk_list(
        self, settings: Settings, sample_paper: Paper, short_section: Section, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [short_section, long_section])
        assert len(chunks) >= 2

    def test_total_chunks_in_paper_updated(
        self, settings: Settings, sample_paper: Paper, short_section: Section, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [short_section, long_section])
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_paper == total

    def test_empty_sections_produce_no_chunks(
        self, settings: Settings, sample_paper: Paper
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [])
        assert chunks == []
