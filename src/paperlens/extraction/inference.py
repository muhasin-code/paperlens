"""Inference wrapper for fine-tuned extraction model."""

import json
import logging
from typing import Literal

from pydantic import ValidationError

from src.paperlens.extraction.models import ExtractionOutput
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.extraction")


class ExtractionInference:
    """Inference wrapper for the fine-tuned extraction model.

    Supports multiple inference modes for different deployment scenarios:
    - hf_api: HuggingFace Inference API (local dev with HF_TOKEN)
    - local_placeholder: Mock output for testing without model access
    - kaggle: Direct model loading (production on Kaggle T4)
    """

    SUPPORTED_MODES = ("hf_api", "local_placeholder", "kaggle")

    def __init__(self, settings: Settings | None = None):
        """Initialize inference wrapper.

        Args:
            settings: Application settings (provides model IDs, inference mode, HF_TOKEN)
        """
        self.settings = settings or Settings()
        self.mode: Literal["hf_api", "local_placeholder", "kaggle"] = (
            self.settings.extraction_inference_mode
        )
        self._model = None
        self._tokenizer = None
        self._hf_client = None

        self._validate_mode()

    def _validate_mode(self) -> None:
        """Validate inference mode."""
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid extraction_inference_mode: {self.mode}. Must be one of {self.SUPPORTED_MODES}"
            )

    async def extract(self, chunk_text: str) -> ExtractionOutput:
        """Extract structured information from a paper chunk.

        Args:
            chunk_text: The paper chunk text to extract from

        Returns:
            ExtractionOutput with all 5 fields (may be null for empty fields)
        """
        if self.mode == "hf_api":
            return await self._infer_with_hf_api(chunk_text)
        elif self.mode == "local_placeholder":
            return self._placeholder_output(chunk_text)
        elif self.mode == "kaggle":
            return await self._infer_local_kaggle(chunk_text)

    async def _infer_with_hf_api(self, chunk_text: str) -> ExtractionOutput:
        """Run inference via HuggingFace Inference API.

        Args:
            chunk_text: Paper chunk text

        Returns:
            Validated ExtractionOutput
        """
        from huggingface_hub import InferenceClient

        if not self.settings.hf_token:
            logger.warning("HF_TOKEN not set, falling back to placeholder")
            return self._placeholder_output(chunk_text)

        try:
            client = InferenceClient(token=self.settings.hf_token)

            prompt = self._build_prompt(chunk_text)

            logger.debug("Calling HF Inference API for extraction...")

            response = client.text_generation(
                prompt,
                max_new_tokens=self.settings.extraction_max_new_tokens or 256,
                temperature=self.settings.extraction_temperature or 0.1,
                model=self.settings.extraction_adapter_repo or self.settings.hf_hub_model_id,
            )

            return self._parse_response(response, chunk_text)

        except Exception as e:
            logger.warning("HF Inference API failed: %s. Falling back to placeholder.", e)
            return self._placeholder_output(chunk_text)

    async def _infer_local_kaggle(self, chunk_text: str) -> ExtractionOutput:
        """Run inference with locally loaded model (Kaggle T4).

        Args:
            chunk_text: Paper chunk text

        Returns:
            Validated ExtractionOutput
        """
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._model is None or self._tokenizer is None:
                logger.info("Loading extraction model locally (Kaggle mode)...")

                base_model_id = "Qwen/Qwen2.5-3B-Instruct"
                adapter_repo = (
                    self.settings.extraction_adapter_repo or self.settings.hf_hub_model_id
                )

                self._tokenizer = AutoTokenizer.from_pretrained(base_model_id)

                self._model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                )

                self._model = PeftModel.from_pretrained(self._model, adapter_repo)
                self._model.eval()

            prompt = self._build_prompt(chunk_text)

            with torch.no_grad():
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.extraction_max_new_tokens or 256,
                    temperature=self.settings.extraction_temperature or 0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_response(response, chunk_text)

        except Exception as e:
            logger.error("Local inference failed: %s", e)
            raise RuntimeError(f"Local extraction inference failed: {e}") from e

    def _build_prompt(self, chunk_text: str) -> str:
        """Build the extraction prompt for the model.

        Args:
            chunk_text: Paper chunk text

        Returns:
            Full prompt with system message and user query
        """
        from src.paperlens.extraction.annotator import DEFAULT_SYSTEM_PROMPT

        user_content = (
            f"Extract structured information from the following ML paper chunk:\n\n{chunk_text}"
        )

        return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_content}"

    def _parse_response(self, response: str, chunk_text: str) -> ExtractionOutput:
        """Parse model response into ExtractionOutput.

        Args:
            response: Raw model output
            chunk_text: Original chunk (for error context)

        Returns:
            Validated ExtractionOutput
        """
        response = response.strip()

        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            logger.warning("No JSON found in response, returning placeholder output")
            return self._placeholder_output(chunk_text)

        json_str = response[json_start:json_end]

        try:
            data = json.loads(json_str)
            return ExtractionOutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("Failed to parse extraction output: %s. Returning placeholder.", e)
            return self._placeholder_output(chunk_text)

    def _placeholder_output(self, chunk_text: str) -> ExtractionOutput:
        """Return a placeholder ExtractionOutput for testing.

        Used when:
        - HF_TOKEN not set and HF API unavailable
        - Running local tests without model access
        - Model produces invalid JSON

        Args:
            chunk_text: Original chunk (ignored in placeholder)

        Returns:
            ExtractionOutput with null fields (valid JSON, no data)
        """
        logger.debug("Returning placeholder ExtractionOutput")
        return ExtractionOutput(
            research_question=None,
            method=None,
            datasets=None,
            metrics=None,
            key_finding=None,
        )

    @classmethod
    def from_mode(
        cls,
        mode: Literal["hf_api", "local_placeholder", "kaggle"],
        settings: Settings | None = None,
    ) -> "ExtractionInference":
        """Factory method to create inference instance with specific mode.

        Args:
            mode: One of "hf_api", "local_placeholder", "kaggle"
            settings: Application settings

        Returns:
            ExtractionInference instance with requested mode
        """
        temp_settings = Settings(extraction_inference_mode=mode) if settings is None else settings
        return cls(temp_settings)


def get_extraction_inference(settings: Settings | None = None) -> ExtractionInference:
    """Factory function for ExtractionInference singleton.

    Args:
        settings: Application settings

    Returns:
        ExtractionInference instance
    """
    return ExtractionInference(settings)
