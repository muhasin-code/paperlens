"""Tests for PromptLoader with mocked YAML files.

Uses tmp_path to write fixture YAML files instead of relying on
actual files in configs/prompts/.
"""

from pathlib import Path

import pytest
import yaml

from src.paperlens.api.prompt_loader import PromptLoader, PromptTemplate
from src.paperlens.settings import Settings


@pytest.fixture
def settings_with_tmp_prompts(tmp_path: Path, monkeypatch) -> Settings:
    """Create Settings with temporary prompts directory."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Prevent .env from overriding our test settings
    monkeypatch.delenv("PROMPTS_DIR", raising=False)
    monkeypatch.delenv("PROMPT_VERSION", raising=False)

    return Settings(prompts_dir=prompts_dir, prompt_version="v1")


@pytest.fixture
def valid_yaml_content() -> dict:
    """Return valid prompt YAML content."""
    return {
        "version": "v1",
        "system_prompt": "You are a helpful assistant.",
        "user_template": "Question: {query}\nContext: {context}\nAnswer:",
        "refusal_message": "I cannot answer from the provided sources.",
    }


@pytest.fixture
def write_yaml_file(tmp_path: Path):
    """Return a helper function to write YAML files."""

    def _write(filename: str, content: dict) -> Path:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        file_path = prompts_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        return file_path

    return _write


class TestPromptLoader:
    def test_load_prompt_success(
        self, settings_with_tmp_prompts, write_yaml_file, valid_yaml_content
    ):
        write_yaml_file("v1.yaml", valid_yaml_content)
        loader = PromptLoader(settings_with_tmp_prompts)
        prompt = loader.load_prompt()

        assert prompt.version == "v1"
        assert "helpful assistant" in prompt.system_prompt
        assert "{query}" in prompt.user_template
        assert "{context}" in prompt.user_template
        assert "cannot answer" in prompt.refusal_message

    def test_load_prompt_caches_result(
        self, settings_with_tmp_prompts, write_yaml_file, valid_yaml_content
    ):
        write_yaml_file("v1.yaml", valid_yaml_content)
        loader = PromptLoader(settings_with_tmp_prompts)

        prompt1 = loader.load_prompt()
        prompt2 = loader.load_prompt()

        assert prompt1 is prompt2
        assert len(loader._cache) == 1

    def test_load_prompt_raises_for_missing_file(self, settings_with_tmp_prompts):
        loader = PromptLoader(settings_with_tmp_prompts)

        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            loader.load_prompt()

    def test_load_prompt_raises_for_missing_fields(
        self, settings_with_tmp_prompts, write_yaml_file
    ):
        incomplete_content = {
            "version": "v1",
            "system_prompt": "You are a helpful assistant.",
        }
        write_yaml_file("v1.yaml", incomplete_content)
        loader = PromptLoader(settings_with_tmp_prompts)

        with pytest.raises(ValueError, match="missing required fields"):
            loader.load_prompt()


class TestPromptTemplate:
    def test_build_prompt_formats_correctly(self):
        template = PromptTemplate(
            version="v1",
            system_prompt="You are a helpful assistant.",
            user_template="Question: {query}\nContext: {context}\nAnswer:",
            refusal_message="I cannot answer from the provided sources.",
        )

        prompt = template.build_prompt(context="Sample context text.", query="What is this?")

        assert "You are a helpful assistant." in prompt
        assert "Question: What is this?" in prompt
        assert "Context: Sample context text." in prompt
        assert "Answer:" in prompt

    def test_build_prompt_handles_special_characters(self):
        template = PromptTemplate(
            version="v1",
            system_prompt="Answer briefly.",
            user_template="Q: {query}\nC: {context}",
            refusal_message="Refuse.",
        )

        prompt = template.build_prompt(
            context="Context with [chunk_001] marker.", query="Test with {special} chars."
        )

        assert "[chunk_001]" in prompt


class TestPromptLoaderWithOverride:
    def test_load_prompt_with_version_override(self, settings_with_tmp_prompts, write_yaml_file):
        # Write v1.yaml
        write_yaml_file(
            "v1.yaml",
            {
                "version": "v1",
                "system_prompt": "v1 prompt",
                "user_template": "Q: {query}\nC: {context}\nA:",
                "refusal_message": "Refuse v1.",
            },
        )

        # Write v2.yaml
        write_yaml_file(
            "v2.yaml",
            {
                "version": "v2",
                "system_prompt": "v2 prompt",
                "user_template": "Q: {query}\nC: {context}\nA:",
                "refusal_message": "Refuse v2.",
            },
        )

        loader = PromptLoader(settings_with_tmp_prompts)

        prompt_v1 = loader.load_prompt("v1")
        prompt_v2 = loader.load_prompt("v2")

        assert prompt_v1.version == "v1"
        assert prompt_v2.version == "v2"
        assert len(loader._cache) == 2
