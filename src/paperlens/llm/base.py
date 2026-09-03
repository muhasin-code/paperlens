"""Abstract base class for LLM provider."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, model: str, options: dict | None = None) -> str:
        """
        Generate a completion (non-streaming).

        Args:
            prompt: The full prompt text.
            model: Model name to use.
            options: Optional generation parameters (temperature, etc.).

        Returns:
            Generated ttext string
        """
        ...

    @abstractmethod
    async def stream(
        self, prompt: str, model: str, options: dict | None = None
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion.

        Args:
            prompt: The full prompt text.
            model: Model name to use.
            options: Optional generation parameters.

        Yields:
            Text tokens as strings.
        """
        ...

    @abstractmethod
    async def is_reachable(self) -> bool:
        """
        Check if the provider is reachable.

        Returns:
            True if the provider is available, False otherwise.
        """
        ...
