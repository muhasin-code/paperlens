"""
Section-aware chunker: splits sections into fixed size token chunks with overlap.

Design decisions:
- tiktoken for accurate token counting (matches emebdding model tokenization).
- Sliding window with configurable overlap to avoid splitting multi-sentence ideas.
- Each chunk carries full provenance: paper ID, section, page range, ordinal position.
"""

import logging

import tiktoken

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.models import Chunk, Section
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlense.parsing")

# cl100k_base is the encoding used by OpenAI embedding models and is
# the standard approximation for most open-source embedding models.
ENCODING_NAME = "cl100k_base"


class SectionAwareChunker:
    """Splits paper sections into fixed-size token chunks with overlap."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunk_size = settings.chunk_size_tokens
        self.overlap = settings.chunk_overlap_tokens
        self.enc = tiktoken.get_encoding(ENCODING_NAME)
        self.logger = logger

    def chunk_section(
        self,
        paper: Paper,
        section: Section,
        section_index: int,
        total_sections: int,
        chunk_offset: int,
    ) -> list[Chunk]:
        """
        Split one section into chunks. Returns a list of Chunk objects.

        Args:
            paper: Source paper metadata.
            section: The section to chunk.
            section_index: Zero-based index of this section witin the paper.
            total_sections: Total sections in this paper.
            chunk_offset: Global chunk index offset (for unique ID generation).
        """
        if not section.text.strip():
            return []

        tokens = self.enc.encode(section.text, disallowed_special=())
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            # Section fits in a single chunk.
            return [
                self._make_chunk(
                    paper=paper,
                    section=section,
                    chunk_index=0,
                    text=section.text,
                    token_count=total_tokens,
                    page_start=section.start_page,
                    page_end=section.end_page,
                    total_chunks_in_section=1,
                    global_offset=chunk_offset,
                )
            ]

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        chunk_idx = 0
        start = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            window_tokens = tokens[start:end]
            window_text = self.enc.decode(window_tokens)

            # Estimate page range by character offset ratio.
            char_start = len(self.enc.decode(tokens[:start]))
            char_total = len(section.text)
            ratio_start = char_start / char_total if char_total > 0 else 0
            ratio_end = (
                min(char_start + len(window_text), char_total) / char_total if char_total > 0 else 1
            )

            page_range = section.end_page - section.start_page + 1
            page_start = section.start_page + int(ratio_start * page_range)
            page_end = section.start_page + int(ratio_end * page_range)
            page_end = max(page_start, page_end)

            chunks.append(
                self._make_chunk(
                    paper=paper,
                    section=section,
                    chunk_index=chunk_idx,
                    text=window_text,
                    token_count=len(window_tokens),
                    page_start=page_start,
                    page_end=page_end,
                    total_chunks_in_section=0,  # updated below
                    global_offset=chunk_offset + chunk_idx,
                )
            )

            chunk_idx += 1
            start += step

        # Update total_chunks_in_section now that we know the count
        total = len(chunks)
        for c in chunks:
            c.total_chunks_in_section = total

        return chunks

    def _make_chunk(
        self,
        paper: Paper,
        section: Section,
        chunk_index: int,
        text: str,
        token_count: int,
        page_start: int,
        page_end: int,
        total_chunks_in_section: int,
        global_offset: int,
    ) -> Chunk:
        """Construct a Chunk with a globally unique ID."""
        padded_index = str(global_offset + 1).zfill(4)
        return Chunk(
            chunk_id=f"{paper.arxiv_id}_chunk_{padded_index}",
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            section_label=section.label,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            text=text.strip(),
            token_count=token_count,
            total_chunks_in_section=total_chunks_in_section,
            total_chunks_in_paper=0,  # updated by pipeline
        )

    def chunk_paper(self, paper: Paper, sections: list[Section]) -> list[Chunk]:
        """Chunk all sections in a paper. Returns a flat list of chunks."""
        all_chunks: list[Chunk] = []
        chunk_offset = 0

        for idx, section in enumerate(sections):
            section_chunks = self.chunk_section(
                paper=paper,
                section=section,
                section_index=idx,
                total_sections=len(sections),
                chunk_offset=chunk_offset,
            )
            all_chunks.extend(section_chunks)
            chunk_offset += len(section_chunks)

        # Update total_chunks_in_paper on every chunk.
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks_in_paper = total

        return all_chunks
