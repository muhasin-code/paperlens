"""
LLM provider abstraction layer.

Supports Ollama (local) and OpenAI-compatible endpoints (Groq, OpenRouter, etc)
via the LLM_PROVIDER environment variable.
"""

from src.paperlens.llm.base import LLMProvider
from src.paperlens.llm.ollama_provider import OllamaProvider
from src.paperlens.llm.openai_compat_provider import OpenAICompatProvider
from src.paperlens.settings import Settings

__all__ = ["LLMProvider", "OllamaProvider", "OpenAICompatProvider", "get_llm_provider"]


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Factory: create LLM provider based on settings.llm_provider."""
    provider_name = settings.llm_provider
    if provider_name == "ollama":
        return OllamaProvider(host=settings.ollama_base_url, model=settings.ollama_model)
    elif provider_name == "openai_compat":
        return OpenAICompatProvider(
            base_url=settings.llm_api_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model or settings.ollama_model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
