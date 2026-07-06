"""
PDF  text extraction and section boundary detection.

Design decisions:
- PyMuPDF (fitz) for text extraction: fast, accurate, no externam dependencies.
- Rule-based section detection: deterministic and debuggable.
- Returns section objects with confidence scores with downstream weighting.
"""

import logging
import re
from pathlib import Path

import fitz

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.models import Section

logger = logging.getLogger("paperlens.parsing")

# Common section headings in ML papers, ordered by priority.
# Each pattern is matched case-insensitively against the first 200 chars of a page.
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*abstract", "abstract"),
    (r"^\s*\d*\.?\s*introduction", "introduction"),
    (r"^\s*\d*\.?\s*related\s+work", "related_work"),
    (r"^\s*\d*\.?\s*background", "background"),
    (r"^\s*\d*\.?\s*method(?:ology)?", "method"),
    (r"^\s*\d*\.?\s*approach", "method"),
    (r"^\s*\d*\.?\s*experiment(?:s|al)?", "experiments"),
    (r"^\s*\d*\.?\s*result(?:s)?", "results"),
    (r"^\s*\d*\.?\s*evaluation", "evaluation"),
    (r"^\s*\d*\.?\s*discussion", "discussion"),
    (r"^\s*\d*\.?\s*conclusion(?:s)?", "conclusion"),
    (r"^\s*\d*\.?\s*references", "references"),
    (r"^\s*\d*\.?\s*acknowledg(?:ement)?", "acknowledgements"),
    (r"^\s*\d*\.?\s*appendix", "appendix"),
]

# Compile once at module load.
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.MULTILINE), label)
    for pattern, label in SECTION_PATTERNS
]


class PdfParser:
    """Extracts text and detects section boundaries from a paper PDF."""

    def __init__(self) -> None:
        self.logger = logger

    def detect_sections(self, pages: list[str]) -> list[Section]:
        """
        Detects section boundaries from extracted page texts.

        Returns a list of section objects covering all pages. Pages before the first detected heading are assigned to 'abstract' (common for title+abstract on page 1).
        """
        if not pages:
            return []

        sections: list[Section] = []
        current_label = "abstract"
        current_start = 1
        current_texts: list[str] = []

        for page_num_0, page_text in enumerate(pages):
            page_num = page_num_0 + 1  # 1-indexed
            first_block = page_text[:300].strip()

            detected_label = self._match_heading(first_block)

            if detected_label is not None and detected_label != current_label:
                # Close the current section.
                if current_texts:
                    sections.append(
                        Section(
                            label=current_label,
                            start_page=current_start,
                            end_page=page_num - 1 if page_num > current_start else page_num,
                            text="\n\n".join(current_texts),
                            confidence=1.0,
                        )
                    )
                current_label = detected_label
                current_start = page_num
                current_texts = [page_text]
            else:
                current_texts.append(page_text)

        # Close the final section.
        if current_texts:
            sections.append(
                Section(
                    label=current_label,
                    start_page=current_start,
                    end_page=len(pages),
                    text="\n\n".join(current_texts),
                    confidence=1.0,
                )
            )

        # Filter out references and appendix - not useful for retrieval.
        filtered = [s for s in sections if s.label not in ("references", "appendix")]

        self.logger.debug(
            "Detected %d sections (%d after filtering) across %d pages",
            len(sections),
            len(filtered),
            len(pages),
        )
        return filtered

    def extract_text_by_page(self, pdf_path: Path) -> list[str]:
        """Extract raw text from a PDF, returning one string per page."""
        pages: list[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        return pages

    def _match_heading(self, text: str) -> str | None:
        """Return the section label if the text starts with a known heading, else None."""
        for pattern, label in _COMPILED_PATTERNS:
            if pattern.search(text):
                return label
        return None

    def parse_paper(self, paper: Paper) -> list[Section]:
        """
        Parse a paper's PDF and return detected sections.

        Returns an empty list if the PDF is missing or unreadable.
        """
        if paper.pdf_path is None:
            self.logger.warning("No PDF path for %s - skipping.", paper.arxiv_id)
            return []

        pdf_path = Path(paper.pdf_path)
        if not pdf_path.exists():
            self.logger.warning("PDF not found: %s - skipping.", pdf_path)
            return []

        try:
            pages = self.extract_text_by_page(pdf_path)
            return self.detect_sections(pages)
        except Exception as exc:
            self.logger.error("Failed to parse %s: %s", paper.arxiv_id, exc)
            return []
