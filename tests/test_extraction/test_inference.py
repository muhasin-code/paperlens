"""Unit tests for ExtractionInference."""

import pytest

from src.paperlens.extraction.inference import ExtractionInference
from src.paperlens.extraction.models import ExtractionOutput
from src.paperlens.settings import Settings


class TestExtractionInference:
    """Tests for ExtractionInference inference modes."""

    def test_placeholder_mode_returns_valid_output(self):
        """Placeholder mode should return valid ExtractionOutput with null fields."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        output = inference._placeholder_output("test chunk text")

        assert isinstance(output, ExtractionOutput)
        assert output.research_question is None
        assert output.method is None
        assert output.datasets is None
        assert output.metrics is None
        assert output.key_finding is None

    def test_placeholder_mode_is_valid_pydantic(self):
        """Placeholder output should be valid Pydantic model."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        output = inference._placeholder_output("test")

        # Should have all required fields
        assert hasattr(output, "to_json")
        json_str = output.to_json()
        assert "research_question" in json_str
        assert "method" in json_str

    def test_invalid_mode_raises_error(self):
        """Invalid inference mode should raise ValueError."""
        settings = Settings(extraction_inference_mode="invalid_mode")

        with pytest.raises(ValueError, match="Invalid extraction_inference_mode"):
            ExtractionInference(settings)

    def test_supported_modes_constant(self):
        """SUPPORTED_MODES should contain expected values."""
        assert "hf_api" in ExtractionInference.SUPPORTED_MODES
        assert "local_placeholder" in ExtractionInference.SUPPORTED_MODES
        assert "kaggle" in ExtractionInference.SUPPORTED_MODES

    def test_build_prompt_includes_system_prompt(self):
        """Built prompt should include system prompt structure."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        prompt = inference._build_prompt("Test chunk content")

        assert "extract structured information" in prompt.lower()
        assert "research_question" in prompt.lower()
        assert "method" in prompt.lower()

    def test_parse_valid_json_response(self):
        """Parsing a valid JSON response should work."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        response = '{"research_question": "Test Q", "method": "Test M", "datasets": ["A"], "metrics": ["acc"], "key_finding": "Test"}'
        output = inference._parse_response(response, "chunk")

        assert output.research_question == "Test Q"
        assert output.method == "Test M"

    def test_parse_response_with_whitespace_and_json(self):
        """Parsing response with extra text before/after JSON should work."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        response = 'Here is the output:\n{"research_question": "Q", "method": null, "datasets": [], "metrics": null, "key_finding": "F"}\nThat\'s it.'
        output = inference._parse_response(response, "chunk")

        assert output.research_question == "Q"
        assert output.key_finding == "F"

    def test_parse_invalid_json_returns_placeholder(self):
        """Parsing invalid JSON should return placeholder output."""
        settings = Settings(extraction_inference_mode="local_placeholder")
        inference = ExtractionInference(settings)

        response = "This is not valid JSON output"
        output = inference._parse_response(response, "chunk")

        assert isinstance(output, ExtractionOutput)
        assert output.research_question is None

    def test_factory_method_from_mode(self):
        """Factory method should create instance with correct mode."""
        inference = ExtractionInference.from_mode("local_placeholder")

        assert inference.mode == "local_placeholder"

    def test_settings_override_mode(self):
        """Settings should override mode parameter."""
        settings = Settings(extraction_inference_mode="hf_api")
        inference = ExtractionInference(settings)

        assert inference.mode == "hf_api"
