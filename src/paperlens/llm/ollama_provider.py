"""Ollama provider implimentation using ollama.AsyncClient."""

import logging
from collections.abc import AsyncIterator

import ollama

from src.paperlens.llm.base import LLMProvider

logger = logging.getLogger("paperlens.llm")


class OllamaProvider(LLMProvider):
    """Ollama provider wrapping ollama.AsyncClient."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "phi4-mini",
    ) -> None:
        self.host = host
        self.model = model
        self._client: ollama.AsyncClient | None = None

    def _get_client(self) -> ollama.AsyncClient:
        """Lazily create the Ollama client."""
        if self._client is None:
            self._client = ollama.AsyncClient(host=self.host)
        return self._client

    async def generate(self, prompt: str, model: str, options: dict | None = None) -> str:
        client = self._get_client()
        resp = await client.generate(
            model=model,
            prompt=prompt,
            options=options or {"temperature": 0.1, "num_predict": 256},
        )
        return resp.get("response", "")

    async def stream(
        self, prompt: str, model: str, options: dict | None = None
    ) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.generate(
            model=model,
            prompt=prompt,
            options=options or {"temperature": 0.1, "num_predict": 256},
            stream=True,
        )
        async for chunk in stream:
            token = chunk.get("response", "")
            if token:
                yield token

    async def is_reachable(self) -> bool:
        try:
            client = self._get_client()
            await client.list()
            return True
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
