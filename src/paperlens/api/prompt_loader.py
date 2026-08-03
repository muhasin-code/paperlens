"""Prompt loader: versioned YAML prompts with in-memory caching."""

import logging
from typing import Any

import yaml

from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.api")


class PromptTemplate:
    """Loaded prompt template with all components."""

    def __init__(
        self,
        version: str,
        system_prompt: str,
        user_template: str,
        refusal_message: str,
    ) -> None:
        self.version = version
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.refusal_message = refusal_message

    def build_prompt(self, context: str, query: str) -> str:
        """Build the full prompt from system + user parts."""
        user_prompt = self.user_template.format(context=context, query=query)
        return f"{self.system_prompt}\n\n{user_prompt}"


class PromptLoader:
    """Loads and caches prompt templates from versioned YAML files."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cache: dict[str, PromptTemplate] = {}

    def load_prompt(self, version: str | None = None) -> PromptTemplate:
        """Load prompt template from YAML file.

        Args:
            version: Override prompt version (uses settings.prompt_version if None)

        Returns:
            PromptTemplate with all prompt components
        """
        ver = version or self.settings.prompt_version

        if ver in self._cache:
            logger.debug("Returning cached prompt version: %s", ver)
            return self._cache[ver]

        # Construct path for the specific version (not cached)
        yaml_path = self.settings.prompts_dir / f"{ver}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        required_fields = ("version", "system_prompt", "user_template", "refusal_message")
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Prompt YAML missing required fields: {missing}")

        template = PromptTemplate(
            version=str(data["version"]),
            system_prompt=str(data["system_prompt"]),
            user_template=str(data["user_template"]),
            refusal_message=str(data["refusal_message"]),
        )

        self._cache[ver] = template
        logger.info("Loaded prompt version: %s from %s", ver, yaml_path)

        return template
