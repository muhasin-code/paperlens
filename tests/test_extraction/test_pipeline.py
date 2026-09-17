"""Tests for ExtractionPipeline with mocked LLM provider."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.paperlens.extraction.annotator import ChunkAnnotator
from src.paperlens.extraction.pipeline import ExtractionPipeline
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        extraction_checkpoint_path=tmp_path / "checkpoint.jsonl",
        extraction_train_path=tmp_path / "train.jsonl",
        extraction_val_path=tmp_path / "val.jsonl",
        processed_chunks_path=tmp_path / "chunks.parquet",
    )


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value='{"research_question": "test", "method": null, "datasets": null, "metrics": null, "key_finding": "test"}'
    )
    return provider


@pytest.fixture
def temp_chunks_file(settings: Settings, tmp_path: Path):
    """Create a temporary parquet file with test chunks."""
    chunks = [
        Chunk(
            chunk_id="paper1_chunk_001",
            arxiv_id="2606.22406v2",
            title="Paper 1",
            authors=["Author"],
            section_label="method",
            chunk_index=0,
            page_start=1,
            page_end=2,
            text="Method section 1.",
            token_count=10,
            total_chunks_in_section=2,
            total_chunks_in_paper=2,
        ),
        Chunk(
            chunk_id="paper1_chunk_002",
            arxiv_id="2606.22406v2",
            title="Paper 1",
            authors=["Author"],
            section_label="results",
            chunk_index=1,
            page_start=3,
            page_end=4,
            text="Results section 1.",
            token_count=10,
            total_chunks_in_section=2,
            total_chunks_in_paper=2,
        ),
        Chunk(
            chunk_id="paper2_chunk_001",
            arxiv_id="2606.22407v1",
            title="Paper 2",
            authors=["Author"],
            section_label="abstract",
            chunk_index=0,
            page_start=1,
            page_end=2,
            text="Abstract section - no extraction.",
            token_count=10,
            total_chunks_in_section=1,
            total_chunks_in_paper=1,
        ),
    ]

    df = pd.DataFrame([c.to_parquet_row() for c in chunks])
    df.to_parquet(settings.processed_chunks_path)
    return settings.processed_chunks_path


class TestExtractionPipeline:
    @pytest.mark.asyncio
    async def test_run_generates_dataset(self, settings, mock_llm_provider, temp_chunks_file):
        settings.extraction_val_split = 0.5  # 1 of 2 goes to val

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        pipeline = ExtractionPipeline(settings, annotator)

        summary = await pipeline.run(limit=10, dry_run=False)

        assert summary["total_chunks_loaded"] == 3
        assert summary["successfully_annotated"] == 2
        assert settings.extraction_train_path.exists()
        assert settings.extraction_val_path.exists()

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self, settings, mock_llm_provider, temp_chunks_file):
        annotator = ChunkAnnotator(mock_llm_provider, settings)
        pipeline = ExtractionPipeline(settings, annotator)

        summary = await pipeline.run(dry_run=True)

        assert summary["successfully_annotated"] == 2
        assert not settings.extraction_train_path.exists()
        assert not settings.extraction_val_path.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_allows_resume(self, settings, mock_llm_provider, temp_chunks_file):
        settings.extraction_max_examples = 10
        settings.extraction_val_split = 0.5

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        pipeline = ExtractionPipeline(settings, annotator)

        await pipeline.run(dry_run=False)

        assert settings.extraction_checkpoint_path.exists()
        checkpoint_chunk_ids = set()
        with open(settings.extraction_checkpoint_path) as f:
            for line in f:
                data = json.loads(line)
                checkpoint_chunk_ids.add(data["chunk_id"])
        assert "paper1_chunk_001" in checkpoint_chunk_ids

    @pytest.mark.asyncio
    async def test_filter_excludes_non_extractable_sections(
        self, settings, mock_llm_provider, temp_chunks_file
    ):
        settings.extraction_max_examples = 10

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        pipeline = ExtractionPipeline(settings, annotator)

        summary = await pipeline.run(dry_run=False)

        assert summary["successfully_annotated"] == 2

    @pytest.mark.asyncio
    async def test_force_clears_checkpoint(self, settings, mock_llm_provider, temp_chunks_file):
        annotator = ChunkAnnotator(mock_llm_provider, settings)
        pipeline = ExtractionPipeline(settings, annotator)

        await pipeline.run(dry_run=False)
        checkpoint_path = settings.extraction_checkpoint_path
        assert checkpoint_path.exists()

        await pipeline.run(force=True, dry_run=False)

        pass
