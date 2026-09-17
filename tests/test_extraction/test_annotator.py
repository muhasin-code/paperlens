"""Tests for ChunkAnnotator with mocked LLM provider."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.paperlens.extraction.annotator import AnnotationError, ChunkAnnotator
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="test_chunk_001",
        arxiv_id="2606.22406v2",
        title="Test Paper",
        authors=["Test Author"],
        section_label="method",
        chunk_index=0,
        page_start=1,
        page_end=2,
        text="This is a test chunk about method.",
        token_count=10,
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )


class TestChunkAnnotator:
    @pytest.mark.asyncio
    async def test_annotate_success(self, settings, mock_llm_provider, sample_chunk):
        mock_llm_provider.generate.return_value = (
            '{"research_question": "Test question", "method": null, '
            '"datasets": null, "metrics": null, "key_finding": "Test finding"}'
        )

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        result = await annotator.annotate(sample_chunk)

        assert result.chunk_id == "test_chunk_001"
        assert result.output.research_question == "Test question"
        assert result.output.key_finding == "Test finding"
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_annotate_with_all_fields(self, settings, mock_llm_provider, sample_chunk):
        mock_llm_provider.generate.return_value = (
            '{"research_question": "How to improve?", "method": "Use layer norm.", '
            '"datasets": ["ImageNet", "CIFAR-10"], "metrics": ["accuracy"], "key_finding": "15% improvement"}'
        )

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        result = await annotator.annotate(sample_chunk)

        assert result.output.research_question == "How to improve?"
        assert result.output.method == "Use layer norm."
        assert result.output.datasets == ["ImageNet", "CIFAR-10"]
        assert result.output.metrics == ["accuracy"]
        assert result.output.key_finding == "15% improvement"

    @pytest.mark.asyncio
    async def test_annotate_invalid_json(self, settings, mock_llm_provider, sample_chunk):
        mock_llm_provider.generate.return_value = "This is not JSON"

        annotator = ChunkAnnotator(mock_llm_provider, settings)

        with pytest.raises(AnnotationError, match="not valid JSON"):
            await annotator.annotate(sample_chunk)

    @pytest.mark.asyncio
    async def test_annotate_empty_response(self, settings, mock_llm_provider, sample_chunk):
        mock_llm_provider.generate.return_value = ""

        annotator = ChunkAnnotator(mock_llm_provider, settings)

        with pytest.raises(AnnotationError, match="Empty response"):
            await annotator.annotate(sample_chunk)

    @pytest.mark.asyncio
    async def test_annotate_missing_fields(self, settings, mock_llm_provider, sample_chunk):
        # Pydantic accepts missing optional fields, filling with None
        mock_llm_provider.generate.return_value = '{"research_question": "Test"}'

        annotator = ChunkAnnotator(mock_llm_provider, settings)
        result = await annotator.annotate(sample_chunk)

        assert result.output.research_question == "Test"
        assert result.output.method is None
        assert result.output.datasets is None
        assert result.output.metrics is None
        assert result.output.key_finding is None


class TestExtractionOutput:
    def test_to_json_excludes_none(self):
        from src.paperlens.extraction.models import ExtractionOutput

        output = ExtractionOutput(
            research_question="Test question",
            method=None,
            datasets=None,
            metrics=None,
            key_finding="Test finding",
        )

        json_str = output.to_json()
        parsed = json.loads(json_str)

        assert "research_question" in parsed
        assert "key_finding" in parsed
        assert "method" not in parsed
        assert "datasets" not in parsed

    def test_all_fields_present(self):
        from src.paperlens.extraction.models import ExtractionOutput

        output = ExtractionOutput(
            research_question="Q",
            method="M",
            datasets=["D"],
            metrics=["acc"],
            key_finding="F",
        )

        json_str = output.to_json()
        assert all(
            f in json_str
            for f in ["research_question", "method", "datasets", "metrics", "key_finding"]
        )
