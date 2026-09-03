"""OpenAI-compatible provider using httpx for remote LLM APIs."""

import logging
from collections.abc import AsyncIterator

import httpx

from src.paperlens.llm.base import LLMProvider

logger = logging.getLogger("paperlens.llm")


class OpenAICompatProvider(LLMProvider):
    """Provider for OpenAI-compatible endpoints (Groq, OpenRouter, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        model: str,
        options: dict | None = None,
    ) -> str:
        client = self._get_client()
        temperature = (options or {}).get("temperature", 0.1)

        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream(
        self,
        prompt: str,
        model: str,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        temperature = (options or {}).get("temperature", 0.1)

        async with client.stream(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line = line[6:]
                if line and line != "[DONE]":
                    try:
                        data = httpx._content.json_decoding.json.loads(line)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except Exception:
                        continue

    async def is_reachable(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except Exception as exc:
            logger.warning("OpenAI-compatible provider health check failed: %s", exc)
            return False
