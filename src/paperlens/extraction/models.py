"""Pydantic models for extraction examples used in QLoRA fine-tuning."""

import json
from typing import Any

from pydantic import BaseModel, Field


class ExtractionOutput(BaseModel):
    """Structured extraction output from a paper chunk."""

    research_question: str | None = Field(
        default=None, description="The core problem or hypothesis the paper addresses"
    )
    method: str | None = Field(
        default=None, description="The proposed approach or algorithm (one paragraph max)"
    )
    datasets: list[str] | None = Field(default=None, description="List of dataset names used")
    metrics: list[str] | None = Field(
        default=None, description="List of evaluation metrics reported"
    )
    key_finding: str | None = Field(
        default=None, description="The main result or contribution in one sentence"
    )

    def to_json(self) -> str:
        """Return JSON string with null fields excluded for cleaner output."""
        return json.dumps(self.model_dump(exclude_none=True))

    def __str__(self) -> str:
        return self.to_json()


class ExtractionExample(BaseModel):
    """A single (chunk, extraction) pair for fine-tuning."""

    chunk_id: str = Field(..., description="Source chunk identifier")
    arxiv_id: str = Field(..., description="Source paper arXiv ID")
    section_label: str = Field(..., description="Section the chunk belongs to")
    input_text: str = Field(..., description="The chunk text to extract from")
    output: ExtractionOutput = Field(..., description="Structured extraction")
    system_prompt: str = Field(..., description="System prompt for the teacher model")

    def to_sft_row(self) -> dict[str, Any]:
        """
        Convert to TRL SFTTrainer compatible format.

        Returns a dict with 'messages' and 'metadata' keys suitable for
        instruction-following fine-tuning with chat templates.
        """
        user_content = f"Extract structured information from the following ML paper chunk:\n\n{self.input_text}"
        assistant_content = self.output.to_json()

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "chunk_id": self.chunk_id,
                "arxiv_id": self.arxiv_id,
                "section_label": self.section_label,
            },
        }

    def to_jsonl(self) -> str:
        """Return a single JSONL line."""
        return json.dumps(self.to_sft_row()) + "\n"
