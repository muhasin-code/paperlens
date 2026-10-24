"""OpenAI-compatible provider using httpx for remote LLM APIs."""

import json
import logging
import time
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
        opts = options or {}
        temperature = opts.get("temperature", 0.1)
        max_tokens = opts.get("max_completion_tokens") or opts.get("num_predict") or 1024

        messages: list[dict[str, str]] = []
        system = opts.get("system")
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        # gpt-oss defaults to medium reasoning, which often spends the whole
        # budget before the answer. "low" is the right setting for grounded QA.
        # Other Groq models reject this field with 400.
        if "gpt-oss" in model:
            body["reasoning_effort"] = opts.get("reasoning_effort", "low")

        response = await client.post(f"{self.base_url}/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        logger.info(
            "Groq finish_reason=%s content_chars=%d model=%s",
            choice.get("finish_reason"),
            len(content),
            model,
        )
        if not content.strip():
            # Reasoning models sometimes return the answer only in `reasoning`.
            content = message.get("reasoning") or ""
            logger.warning("Groq content empty; using reasoning field (%d chars)", len(content))
        return content

    async def stream(
        self,
        prompt: str,
        model: str,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from OpenAI-compatible API.

        Matches generate(): system/user roles, max_completion_tokens from num_predict,
        reasoning_effort=low only for gpt-oss models. Yields delta.constant only.
        """

        client = self._get_client()
        opts = options or {}
        temperature = opts.get("temperature", 0.1)
        max_tokens = opts.get("max_completion_tokens") or opts.get("num_predict") or 1024

        messages: list[dict[str, str]] = []
        system = opts.get("system")
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if "gpt-oss" in model:
            body["reasoning_effort"] = opts.get("reasoning_effort", "low")

        stream_start = time.perf_counter()
        first_chunk_logged = False

        try:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=body
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        line = line[6:]
                    if line and line != "[DONE]":
                        try:
                            data = json.loads(line)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    if not first_chunk_logged:
                                        ttft_ms = (time.perf_counter() - stream_start) * 1000
                                        logger.info(
                                            "OpenAI-compat TTFT %.0f ms model=%s", ttft_ms, model
                                        )
                                        first_chunk_logged = True
                                    yield delta["content"]
                        except Exception:
                            continue
        except Exception as exc:
            logger.error("OpenAI-compatible streaming error: %s", exc)
            raise

    async def is_reachable(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as exc:
            logger.warning("OpenAI-compatible provider health check failed: %s", exc)
            return False
