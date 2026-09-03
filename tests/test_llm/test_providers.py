"""Tests for LLM provider abstractions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.paperlens.llm.ollama_provider import OllamaProvider
from src.paperlens.llm.openai_compat_provider import OpenAICompatProvider


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_is_reachable_returns_true_when_ollama_responds(self):
        with patch("src.paperlens.llm.ollama_provider.ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.return_value = {"models": [{"name": "phi4-mini"}]}
            mock_client_class.return_value = mock_client

            provider = OllamaProvider(host="http://localhost:11434", model="phi4-mini")
            result = await provider.is_reachable()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_reachable_returns_false_on_error(self):
        with patch("src.paperlens.llm.ollama_provider.ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.side_effect = ConnectionError("Connection refused")
            mock_client_class.return_value = mock_client

            provider = OllamaProvider(host="http://localhost:11434", model="phi4-mini")
            result = await provider.is_reachable()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_returns_text(self):
        with patch("src.paperlens.llm.ollama_provider.ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "test answer"}
            mock_client_class.return_value = mock_client

            provider = OllamaProvider(host="http://localhost:11434", model="phi4-mini")
            result = await provider.generate("test prompt", "phi4-mini")
            assert result == "test answer"
            mock_client.generate.assert_called_once()


class TestOpenAICompatProvider:
    @pytest.mark.asyncio
    async def test_is_reachable_returns_true_when_api_responds(self):
        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            api_key="test-key",
            model="test-model",
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.is_reachable()
            assert result is True

    @pytest.mark.asyncio
    async def test_generate_returns_text(self):
        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            api_key="test-key",
            model="test-model",
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "test response"}}]}

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.generate("test prompt", "test-model")
            assert result == "test response"
