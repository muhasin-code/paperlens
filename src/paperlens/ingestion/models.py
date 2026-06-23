"""
Pydantic model for arXiv paper metadata.

Paper is the canonical data strucure for the ingestion and retrieval pipeline.
Serialises to/from JSONL lines via to_jasonl() / from_jasonl()
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Represents one arXiv paper and its local storage state."""

    arxiv_id: str
    """Short arXiv id including version suffix, e.g. '2301.07597v2'."""

    title: str
    abstract: str
    authors: list[str]
    categories: list[str]

    published: datetime
    """Original submission date (UTC)."""

    updated: datetime
    """Date of last revision (UTC)."""

    pdf_url: str
    """Canonical PDF URL from the arXiv API."""

    pdf_path: str | None = None
    """Absolute local path to the downloaded PDF. None if not yet downloaded."""

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    """Timestamp of when this record was written to metadate.jsonl."""

    def to_jsonl(self) -> str:
        """Serialise to a sinle JSONL line (JSON + newline) for appending."""
        return self.model_dump_json() + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "Paper":
        """Parse one JSONL line back into a Paper instance."""
        return cls.model_validate_json(line.strip())
