"""
Pydantic models for parsed paper sections and chunks.

Section: a contiguous block of text under one heading (e.g., "3.2 Method").
Chunk: a fixed-sized token window within a section, with full provenance metadata.
"""

from pydantic import BaseModel


class Section(BaseModel):
    """A detected section within a paper."""

    label: str
    """Human-readable section label, e.g. 'abstract', 'introduction', 'method', 'results'."""

    start_page: int
    """1-indexed page number where this section begins."""

    end_page: int
    """1-indexed page number where this section ends (inclusive)."""

    text: str
    """Full extracted text for this section."""

    confidence: float = 1.0
    """Detection confidence: 1.0 for explicit headings, lower for heuristic matches."""


class Chunk(BaseModel):
    """A fixed-size token chunk with full provenance for retrieval and citation."""

    chunk_id: str
    """Globally unique ID: '{arxiv_id}_chunk_{zero_padded_index}', e.g. '2301.07597v2_chunk_0001'."""

    arxiv_id: str
    """Source paper arXiv ID."""

    title: str
    """Source paper title (denormalized for citation display)."""

    authors: list[str]
    """Source paper authors (denormalized for citation display)."""

    section_label: str
    """Section this chunk belongs to (matches Section.label)."""

    chunk_index: int
    """Zero-based ordinal index of this chunk within its section."""

    page_start: int
    """1-indexed starting page of this chunk's text."""

    page_end: int
    """1-indexed ending page of this chunk's text."""

    text: str
    """The actual chunk text (≤ chunk_size_tokens tokens)."""

    token_count: int
    """Actual token count of `text` as measured by tiktoken."""

    total_chunks_in_section: int
    """Total chunks in this section (for UI: "chunk 3 of 12 in Method")."""

    total_chunks_in_paper: int
    """Total chunks across all sections in this paper."""

    def to_parquet_row(self) -> dict:
        """Convert to a flat dict suitable for Parquet serialization."""
        return self.model_dump()

    @classmethod
    def from_parquet_row(cls, row: dict) -> "Chunk":
        """Reconstruct a Chunk from a Parquet row dict."""
        return cls.model_validate(row)
