"""Ollama provider implimentation using ollama.AsyncClient."""

import logging
import time
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
        opts = options or {}
        system = opts.get("system")
        prompt_text = f"{system}\n\n{prompt}" if system else prompt
        resp = await client.generate(
            model=model,
            prompt=prompt_text,
            options={
                "temperature": opts.get("temperature", 0.1),
                "num_predict": opts.get("num_predict", 1024),
            },
        )
        return resp.get("response", "")

    async def stream(
        self, prompt: str, model: str, options: dict | None = None
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama with TTFT logging.

        Default num_predict is 1024 (matching generate()). Logs first chunk TTFT.
        """

        client = self._get_client()
        opts = options or {}
        system = opts.get("system")
        prompt_text = f"{system}\n\n{prompt}" if system else prompt

        stream_start = time.perf_counter()
        first_chunk_logged = False

        stream = await client.generate(
            model=model,
            prompt=prompt_text,
            options={
                "temperature": opts.get("temperature", 0.1),
                "num_predict": opts.get("num_predict", 1024),
            },
            stream=True,
        )
        async for chunk in stream:
            token = chunk.get("response", "")
            if token:
                if not first_chunk_logged:
                    ttft_ms = (time.perf_counter() - stream_start) * 1000
                    logger.info("Ollama TTFT %.0f ms model=%s", ttft_ms, model)
                    first_chunk_logged = True
                yield token

    async def is_reachable(self) -> bool:
        try:
            client = self._get_client()
            await client.list()
            return True
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
