"""Chunk annotation using LLM provider for structured extraction."""

import asyncio
import json
import logging

from pydantic import ValidationError

from src.paperlens.extraction.models import ExtractionExample, ExtractionOutput
from src.paperlens.llm.base import LLMProvider
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.extraction")

DEFAULT_SYSTEM_PROMPT = """You are a research assistant specialized in extracting structured information from machine learning papers.

From the provided paper chunk, extract the following fields and return ONLY a JSON object with these keys (all nullable):

- research_question: The core problem or hypothesis the paper addresses
- method: The proposed approach or algorithm (max one paragraph)
- datasets: List of dataset names used (empty array if none)
- metrics: List of evaluation metrics reported (empty array if none)
- key_finding: The main result or contribution in one sentence

If a field is not present or not useful, set it to null. Do not hallucinate information. Do not add any prose - only return the JSON object.

Example output:
{"research_question": "How to improve...", "method": "...", "datasets": ["..."], "metrics": ["..."], "key_finding": "..."}"""

DEFAULT_RATE_LIMIT_DELAY = 10  # seconds
DEFAULT_BACKOFF_BASE = 60  # seconds for rate limit retry


class AnnotationError(Exception):
    """Raised when annotation fails after retries."""

    def __init__(self, chunk_id: str, reason: str, original_error: Exception | None = None):
        self.chunk_id = chunk_id
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Annotation failed for {chunk_id}: {reason}")


class ChunkAnnotator:
    """Annotates a chunk using the LLM provider with retry and validation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        system_prompt: str | None = None,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ):
        self.llm_provider = llm_provider
        self.settings = settings
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.rate_limit_delay = rate_limit_delay
        self.backoff_base = backoff_base

    async def _call_with_retry(self, prompt: str, model: str, max_retries: int = 5) -> str:
        """Call LLM with exponential backoff on rate limit errors."""
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                result = await self.llm_provider.generate(
                    prompt=prompt,
                    model=model,
                    options={"temperature": 0.1, "num_predict": 1024},
                )
                return result.strip()
            except Exception as exc:
                last_error = exc
                error_str = str(exc).lower()

                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    delay = self.backoff_base * (2**attempt)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d), sleeping %.0fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self.rate_limit_delay)

        raise AnnotationError(
            chunk_id="unknown",
            reason=f"Failed after {max_retries} retries",
            original_error=last_error,
        )

    def _parse_json_safely(self, text: str, chunk_id: str) -> ExtractionOutput:
        """Parse JSON from LLM response, handling common failures."""
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise AnnotationError(
                chunk_id=chunk_id,
                reason="LLM output is not valid JSON",
            ) from None

        try:
            output = ExtractionOutput(**data)
            return output
        except ValidationError as e:
            raise AnnotationError(
                chunk_id=chunk_id,
                reason=f"Output does not match extraction schema: {e}",
            ) from None

    async def annotate(self, chunk: Chunk, model: str | None = None) -> ExtractionExample:
        """
        Annotate a single chunk with structured extraction.

        Args:
            chunk: The Chunk to annotate
            model: Model override (uses settings.llm_model or settings.ollama_model)

        Returns:
            ExtractionExample with the annotated data

        Raises:
            AnnotationError: If annotation fails after retries
        """
        chunk_id = chunk.chunk_id
        model = model or self.settings.llm_model or self.settings.ollama_model

        prompt = f"{self.system_prompt}\n\nExtract structured information from the following ML paper chunk:\n\n{chunk.text}"

        logger.debug("Annotating %s with %s", chunk_id, model)

        try:
            response = await self._call_with_retry(prompt, model)

            if not response:
                raise AnnotationError(chunk_id=chunk_id, reason="Empty response from LLM")

            output = self._parse_json_safely(response, chunk_id)

            return ExtractionExample(
                chunk_id=chunk_id,
                arxiv_id=chunk.arxiv_id,
                section_label=chunk.section_label,
                input_text=chunk.text,
                output=output,
                system_prompt=self.system_prompt,
            )

        except AnnotationError:
            raise
        except Exception as exc:
            raise AnnotationError(chunk_id=chunk_id, reason=str(exc), original_error=exc) from exc
