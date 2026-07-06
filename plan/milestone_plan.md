# Active Milestone Plan

> **Workflow:** This file holds **one milestone at a time**. When finished, archive it as a new Markdown file under `plan/completed-milestones/phase-1-core-rag-pipeline/`, add a summary row to [`plan/completed_milestones.md`](completed_milestones.md), then replace this file with the next milestone from [`plan/blueprint.md`](blueprint.md).

---

## Milestone 1.2 — Document Parsing & Section-Aware Chunking

| Field | Value |
|---|---|
| **Phase** | 1 — Core RAG Pipeline |
| **Blueprint ref** | [`plan/blueprint.md`](blueprint.md) → Milestone 1.2 |
| **Prerequisites** | Milestone 1.1 complete; `data/raw/metadata.jsonl` has ≥200 records; `data/raw/pdfs/` has ≥200 PDFs; `.venv` active; `requirements.txt` installed |
| **Estimated time** | 5–7 hours (includes PDF parsing run over full corpus) |
| **Release tag** | None |

### Goal

Transform the raw PDFs produced by Milestone 1.1 into a structured, section-aware chunk
dataset suitable for embedding and retrieval. Each chunk must carry enough metadata
(section label, paper ID, position) that downstream retrieval (Phase 1.3) and citation
enforcement (Phase 2.4) can point a user back to the exact location in the source paper.

**End state:** `data/processed/chunks.parquet` with ~15,000–25,000 chunks; every chunk
tagged with its source paper, section label, and ordinal position; `make parse` safely
re-runnable and idempotent.

### Deliverables Checklist

- [X] `requirements.txt` updated with `tiktoken` (tokenizer for accurate token counting)
- [X] `src/paperlens/parsing/__init__.py`
- [X] `src/paperlens/parsing/models.py` — `Section` and `Chunk` Pydantic models
- [X] `src/paperlens/parsing/pdf_parser.py` — `PdfParser` class: extract text per page, detect section boundaries (with configurable HEADING_CHAR_LIMIT)
- [X] `src/paperlens/parsing/chunker.py` — `SectionAwareChunker` class: reads chunk_size_tokens and chunk_overlap_tokens from Settings
- [X] `src/paperlens/parsing/pipeline.py` — `ParsingPipeline` class: orchestrates parse → chunk → persist with idempotency
- [X] `scripts/parse.py` — CLI entry point with `--dry-run`, `--limit`, `--force`
- [X] `Makefile` — `parse` and `parse-dry` targets added
- [X] `tests/test_parsing/__init__.py` and `tests/test_parsing/test_chunker.py`
- [X] Parsing executed: `data/processed/chunks.parquet` has ≥10,000 chunks
- [X] `docs/chunking.md` — chunking strategy rationale, section detection approach, sample chunk schema
- [X] All changes committed and pushed (`git push`)

---

## Step 1 — Update `requirements.txt`

### Why tiktoken

Accurate token counting is critical for the 600-token chunk target. Naive whitespace or
character-based splitting produces chunks of wildly varying token counts, which degrades
embedding quality and can blow out the LLM context window in Phase 1.4. `tiktoken` is the
tokenizer used by OpenAI models and is the standard for approximate token counting in
open-source RAG pipelines. It has no heavy transitive dependencies.

### Actions

Add the following to `requirements.txt` under a new "Parsing" section:

```text
# ── Document Parsing (Phase 1) ────────────────────────────────────────────────
tiktoken>=0.7.0                 # accurate token counting for chunk sizing
```

Install it:

```bash
source .venv/bin/activate
pip install tiktoken
python -c "import tiktoken; print(tiktoken.__version__)"
```

### Verification

- [ ] `tiktoken>=0.7.0` is in `requirements.txt`
- [ ] `python -c "import tiktoken"` succeeds inside `.venv`

---

## Step 2 — Create Parsing Package Skeleton

### Actions

```bash
mkdir -p src/paperlens/parsing
mkdir -p tests/test_parsing
touch tests/test_parsing/__init__.py
```

Create `src/paperlens/parsing/__init__.py`:

```python
"""Document parsing and section-aware chunking pipeline."""
```

### Verification

- [ ] `src/paperlens/parsing/` directory exists
- [ ] `src/paperlens/parsing/__init__.py` exists
- [ ] `tests/test_parsing/__init__.py` exists

---

## Step 3 — Define Parsing Models

### Why explicit models

The `Chunk` model is the central data structure for the entire retrieval pipeline. Every
downstream component — embedding (1.3), vector storage (1.3), BM25 indexing (2.1),
reranking (2.3), citation enforcement (2.4) — consumes chunks. Defining the schema now
as a Pydantic model means all downstream code shares one validated contract.

### Actions

Create `src/paperlens/parsing/models.py`:

```python
"""Pydantic models for parsed paper sections and chunks.

Section: a contiguous block of text under one heading (e.g. "3.2 Method").
Chunk: a fixed-size token window within a section, with full provenance metadata.
"""

from pydantic import BaseModel, Field


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
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.parsing.models import Chunk, Section
s = Section(label='method', start_page=3, end_page=5, text='We propose...')
c = Chunk(
    chunk_id='2401.00001v1_chunk_0001',
    arxiv_id='2401.00001v1',
    title='Test Paper',
    authors=['Alice'],
    section_label='method',
    chunk_index=0,
    page_start=3,
    page_end=3,
    text='We propose a novel method.',
    token_count=6,
    total_chunks_in_section=5,
    total_chunks_in_paper=25,
)
print('Models OK')
"
```

---

## Step 4 — Implement PDF Parser with Section Detection

### Section detection strategy

Academic ML papers follow a predictable structure. The parser uses a two-pass approach:

1. **Heading detection** — match common section heading patterns (numbered "1. Introduction",
   uppercase "METHOD", title case "Results and Analysis") against the first text block on each page.
2. **Fallback heuristic** — if no heading is detected, assign the page to the previous section.

This is intentionally rule-based rather than ML-based: it is deterministic, debuggable, and
fast on CPU. The `confidence` field on `Section` lets downstream code weight explicit-heading
sections higher if needed.

### Configurable parameters

The `HEADING_CHAR_LIMIT` is a module-level constant (default: 300). This value is stable
for ML papers and does not need to be configurable per-run. It is documented here for
transparency.

### Actions

Create `src/paperlens/parsing/pdf_parser.py`:

```python
"""PDF text extraction and section boundary detection.

Design decisions:
- PyMuPDF (fitz) for text extraction: fast, accurate, no external dependencies.
- Rule-based section detection: deterministic and debuggable.
- Returns Section objects with confidence scores for downstream weighting.
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.models import Section

logger = logging.getLogger("paperlens.parsing")

# Common section headings in ML papers, ordered by priority.
# Each pattern is matched case-insensitively against the first N chars of a page.
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*abstract", "abstract"),
    (r"^\s*\d*\.?\s*introduction", "introduction"),
    (r"^\s*\d*\.?\s*related\s+work", "related_work"),
    (r"^\s*\d*\.?\s*background", "background"),
    (r"^\s*\d*\.?\s*method(?:ology)?", "method"),
    (r"^\s*\d*\.?\s*approach", "method"),
    (r"^\s*\d*\.?\s*model", "method"),
    (r"^\s*\d*\.?\s*experiment(?:s|al)?", "experiments"),
    (r"^\s*\d*\.?\s*result(?:s)?", "results"),
    (r"^\s*\d*\.?\s*evaluation", "evaluation"),
    (r"^\s*\d*\.?\s*discussion", "discussion"),
    (r"^\s*\d*\.?\s*conclusion(?:s)?", "conclusion"),
    (r"^\s*\d*\.?\s*references", "references"),
    (r"^\s*\d*\.?\s*acknowledg(?:ement)?", "acknowledgments"),
    (r"^\s*\d*\.?\s*appendix", "appendix"),
]

# Compile once at module load.
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.MULTILINE), label)
    for pattern, label in SECTION_PATTERNS
]

# Number of characters to examine for heading detection.
HEADING_CHAR_LIMIT = 300


class PdfParser:
    """Extracts text and detects section boundaries from a paper PDF."""

    def __init__(self) -> None:
        self.logger = logger

    def extract_text_by_page(self, pdf_path: Path) -> list[str]:
        """Extract text from a PDF, returning one string per page (0-indexed list)."""
        pages: list[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                pages.append(text)
        return pages

    def detect_sections(self, pages: list[str]) -> list[Section]:
        """Detect section boundaries from extracted page texts.

        Returns a list of Section objects covering all pages. Pages before the
        first detected heading are assigned to 'abstract' (common for title+abstract
        on page 1).
        """
        if not pages:
            return []

        sections: list[Section] = []
        current_label = "abstract"
        current_start = 1
        current_texts: list[str] = []

        for page_num_0, page_text in enumerate(pages):
            page_num = page_num_0 + 1  # 1-indexed
            first_block = page_text[:HEADING_CHAR_LIMIT].strip()

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

        # Filter out references and appendix — not useful for retrieval.
        filtered = [s for s in sections if s.label not in ("references", "appendix")]

        self.logger.debug(
            "Detected %d sections (%d after filtering) across %d pages",
            len(sections),
            len(filtered),
            len(pages),
        )
        return filtered

    def _match_heading(self, text: str) -> str | None:
        """Return the section label if the text starts with a known heading, else None."""
        for pattern, label in _COMPILED_PATTERNS:
            if pattern.search(text):
                return label
        return None

    def parse_paper(self, paper: Paper) -> list[Section]:
        """Parse a paper's PDF and return detected sections.

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
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.parsing.pdf_parser import PdfParser
parser = PdfParser()
print('Parser imported OK')
"
```

---

## Step 5 — Implement Section-Aware Chunker (Settings-Integrated)

### Chunking strategy

- Tokenize each section's text using `tiktoken` (cl100k_base encoding — the standard for
  embedding models).
- Walk through the token stream with a sliding window: `chunk_size_tokens` tokens per step,
  `chunk_overlap_tokens` tokens of overlap with the previous chunk.
- Map each token window back to its source page range by tracking character offsets
  during tokenization.
- Assign a globally unique `chunk_id` so retrieval results can be traced back to a
  specific chunk.

The overlap ensures that ideas spanning a chunk boundary are fully captured in at least
one chunk — critical for retrieval quality on multi-sentence claims.

### Actions

Create `src/paperlens/parsing/chunker.py`:

```python
"""Section-aware chunker: splits sections into fixed-size token chunks with overlap.

Design decisions:
- tiktoken for accurate token counting (matches embedding model tokenization).
- Sliding window with configurable overlap to avoid splitting multi-sentence ideas.
- Each chunk carries full provenance: paper ID, section, page range, ordinal position.
"""

import logging
from pathlib import Path

import tiktoken

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.models import Chunk, Section
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.parsing")

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
        """Split one section into chunks. Returns a list of Chunk objects.

        Args:
            paper: Source paper metadata.
            section: The section to chunk.
            section_index: Zero-based index of this section within the paper.
            total_sections: Total sections in this paper.
            chunk_offset: Global chunk index offset (for unique ID generation).
        """
        if not section.text.strip():
            return []

        tokens = self.enc.encode(section.text)
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
            ratio_end = min(char_start + len(window_text), char_total) / char_total if char_total > 0 else 1

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

        # Update total_chunks_in_section now that we know the count.
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
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.parsing.chunker import SectionAwareChunker
settings = Settings()
chunker = SectionAwareChunker(settings)
print(f'Chunker OK: size={chunker.chunk_size}, overlap={chunker.overlap}')
"
```

---

## Step 6 — Implement Parsing Pipeline (Orchestration + Persistence)

### Why a separate pipeline class

The pipeline ties together parsing, chunking, and persistence with idempotency. Separating
this from the parser and chunker keeps each class focused on one responsibility and makes
the pipeline testable with mocked parser/chunker.

### Actions

Create `src/paperlens/parsing/pipeline.py`:

```python
"""Parsing pipeline: orchestrates PDF parsing, chunking, and Parquet persistence.

Design decisions:
- Idempotent: existing chunk output is loaded; already-processed papers are skipped
  unless --force is passed.
- Parquet for persistence: columnar, compressed, fast for batch reads in Phase 1.3.
- Per-paper progress logging so long runs are monitorable.
"""

import logging
from pathlib import Path

import pandas as pd

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.chunker import SectionAwareChunker
from src.paperlens.parsing.models import Chunk
from src.paperlens.parsing.pdf_parser import PdfParser
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.parsing")


class ParsingPipeline:
    """Orchestrates the full parse-and-chunk pipeline over the paper corpus."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = PdfParser()
        self.chunker = SectionAwareChunker(settings)
        self.logger = logger

    def _load_existing_chunk_ids(self) -> set[str]:
        """Return the set of chunk_ids already in the output Parquet file."""
        path = self.settings.processed_chunks_path
        if not path.exists():
            return set()
        try:
            df = pd.read_parquet(path, columns=["chunk_id"])
            return set(df["chunk_id"].tolist())
        except Exception as exc:
            self.logger.warning("Could not read existing chunks: %s", exc)
            return set()

    def _load_papers_with_pdfs(self) -> list[Paper]:
        """Load papers from metadata.jsonl that have a downloaded PDF."""
        if not self.settings.metadata_path.exists():
            self.logger.error("Metadata file not found: %s", self.settings.metadata_path)
            return []

        papers: list[Paper] = []
        with self.settings.metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = Paper.from_jsonl(line)
                    if paper.pdf_path is not None:
                        papers.append(paper)
                except Exception as exc:
                    self.logger.warning("Could not parse metadata line: %s", exc)

        self.logger.info("Loaded %d papers with PDFs.", len(papers))
        return papers

    def _append_chunks_to_parquet(self, chunks: list[Chunk]) -> None:
        """Append chunks to the Parquet file (creates if absent)."""
        if not chunks:
            return

        path = self.settings.processed_chunks_path
        path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([c.to_parquet_row() for c in chunks])

        if path.exists():
            existing_df = pd.read_parquet(path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)

    def run(self, dry_run: bool = False, limit: int | None = None, force: bool = False) -> dict:
        """Execute the full parsing pipeline.

        Args:
            dry_run: If True, parse and chunk but do not write to disk.
            limit: If set, only process the first N papers (for smoke tests).
            force: If True, re-process papers even if their chunks already exist.

        Returns:
            Dict of stats: mode, papers_processed, papers_skipped, total_chunks,
            avg_chunks_per_paper, output_path.
        """
        self.logger.info(
            "=== PaperLens Parsing Start | dry_run=%s | limit=%s | force=%s ===",
            dry_run, limit, force,
        )

        papers = self._load_papers_with_pdfs()
        if limit is not None:
            papers = papers[:limit]

        existing_ids = set() if force else self._load_existing_chunk_ids()

        papers_processed = 0
        papers_skipped = 0
        all_new_chunks: list[Chunk] = []

        for paper in papers:
            # Check if any chunks from this paper already exist.
            paper_prefix = f"{paper.arxiv_id}_chunk_"
            has_existing = any(cid.startswith(paper_prefix) for cid in existing_ids)

            if has_existing and not force:
                self.logger.debug("Skip (already parsed): %s", paper.arxiv_id)
                papers_skipped += 1
                continue

            sections = self.parser.parse_paper(paper)
            if not sections:
                self.logger.warning("No sections detected for %s — skipping.", paper.arxiv_id)
                papers_skipped += 1
                continue

            chunks = self.chunker.chunk_paper(paper, sections)
            all_new_chunks.extend(chunks)
            papers_processed += 1
            self.logger.debug(
                "Parsed %s: %d sections → %d chunks", paper.arxiv_id, len(sections), len(chunks)
            )

        if not dry_run and all_new_chunks:
            self._append_chunks_to_parquet(all_new_chunks)

        avg_chunks = (
            len(all_new_chunks) / papers_processed if papers_processed > 0 else 0.0
        )

        self.logger.info(
            "=== Parsing Complete | processed=%d | skipped=%d | new_chunks=%d | avg=%.1f ===",
            papers_processed, papers_skipped, len(all_new_chunks), avg_chunks,
        )

        return {
            "mode": "dry_run" if dry_run else "full",
            "papers_processed": papers_processed,
            "papers_skipped": papers_skipped,
            "total_chunks": len(all_new_chunks),
            "avg_chunks_per_paper": round(avg_chunks, 1),
            "output_path": str(self.settings.processed_chunks_path),
        }
```

### Verification

```bash
source .venv/bin/activate
python -c "
from src.paperlens.parsing.pipeline import ParsingPipeline
from src.paperlens.settings import get_settings
pipeline = ParsingPipeline(get_settings())
print('Pipeline imported OK')
"
```

---

## Step 7 — Create CLI Script and Update Makefile

### 7a — `scripts/parse.py`

```python
#!/usr/bin/env python3
"""CLI entry point for the PaperLens parsing pipeline.

Examples:
    python scripts/parse.py                     # full parse using .env defaults
    python scripts/parse.py --dry-run           # preview without writing
    python scripts/parse.py --limit 5           # parse only 5 papers (smoke test)
    python scripts/parse.py --force             # re-parse all papers
    make parse                                  # same as first option
    make parse-dry                              # same as --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.parsing.pipeline import ParsingPipeline
from src.paperlens.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PDFs into section-aware chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but do not write to disk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N papers (for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse papers even if their chunks already exist.",
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = ParsingPipeline(settings)
    stats = pipeline.run(dry_run=args.dry_run, limit=args.limit, force=args.force)

    print("\n--- Parsing Stats ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

### 7b — Update Makefile

Add the following targets after the existing `ingest-dry` target:

```makefile
# ─── Parsing ────────────────────────────────────────────────────────────────

parse:
	@echo "→ Parsing PDFs into section-aware chunks (~30–60 min for 500 papers)..."
	python scripts/parse.py

parse-dry:
	@echo "→ Dry run: parsing first 5 papers, no writes..."
	python scripts/parse.py --dry-run --limit 5
```

Also update `make help`:

```makefile
	@echo "  make parse        Run full PDF parsing pipeline"
	@echo "  make parse-dry    Preview parsing on 5 papers (no writes)"
```

### Verification

```bash
source .venv/bin/activate
python scripts/parse.py --help   # should print usage without error
make parse-dry                   # parses 5 papers; expect a few seconds
```

The dry-run should show `papers_processed: 5`, `total_chunks > 0`, and `mode: dry_run`.

---

## Step 7b — Post-Implementation Bug Fix: PdfParser Method Mismatch

### Symptom observed

After completing Step 7, running `make parse-dry` produced:

```text
Failed to parse 2606.24884v1: 'PosixPath' object is not iterable
No sections detected for 2606.24884v1 - skipping.
...
papers_processed: 0
papers_skipped: 5
total_chunks: 0
```

### Root-cause analysis

There are **two compounding bugs** in `src/paperlens/parsing/pdf_parser.py`:

1. **Wrong argument type passed to `extract_text_by_page()`.**

   In `PdfParser.parse_paper()` (lines 134–136) the call is:

   ```python
   pages = self.extract_text_by_page(pdf_path)   # pdf_path is a PosixPath
   return self.detect_sections(pages)
   ```

   But the method is defined as:

   ```python
   def extract_text_by_page(self, pages: list[str]) -> list[Section]:
   ```

   It expects a `list[str]` (raw page texts), but `parse_paper()` passes it a `Path` object.
   When the method body iterates with `for page_num_0, page_text in enumerate(pages)`,
   Python attempts to iterate the `PosixPath`, which is not iterable — raising:

   ```
   TypeError: 'PosixPath' object is not iterable
   ```

2. **`detect_sections()` method does not exist.**

   Even if bug #1 were fixed, `parse_paper()` calls `self.detect_sections(pages)`
   which is never defined anywhere in the class. It would raise `AttributeError`.

3. **Misnamed method with wrong responsibility.**

   The existing `extract_text_by_page` method actually implements **section detection**
   logic (heading matching, section grouping) and returns `list[Section]`. There is **no
   PyMuPDF (`fitz`) text extraction step at all**, even though the module docstring
   explicitly lists it as a design decision.

### How to rectify

Split the responsibilities into **two properly typed methods**, matching what
`parse_paper()` expects:

| Current (broken) | Correct implementation |
|---|---|
| `extract_text_by_page(pdf_path)` — signature says `list[str]`, body does section detection | `extract_text_by_page(pdf_path: Path) -> list[str]` — opens PDF with `fitz`, returns one text string per page |
| `detect_sections(pages)` — does not exist | `detect_sections(pages: list[str]) -> list[Section]` — runs heading-matching logic on raw page texts |

Concrete changes for `src/paperlens/parsing/pdf_parser.py`:

1. Add `import fitz` at the top of the file (next to existing imports).
2. Rename the existing `extract_text_by_page` method (lines 51–110) to **`detect_sections`**.
   Its logic (heading matching, section grouping, references filtering) is exactly what
   `detect_sections` should do. Update its docstring accordingly.
3. Add a new **`extract_text_by_page`** method that reads the PDF:

   ```python
   def extract_text_by_page(self, pdf_path: Path) -> list[str]:
       """Extract raw text from a PDF, returning one string per page."""
       pages: list[str] = []
       with fitz.open(pdf_path) as doc:
           for page in doc:
               pages.append(page.get_text("text"))
       return pages
   ```

4. Verify `parse_paper()` already calls these two methods correctly:

   ```python
   try:
       pages = self.extract_text_by_page(pdf_path)   # list[str]
       return self.detect_sections(pages)             # list[Section]
   except Exception as exc:
       self.logger.error("Failed to parse %s: %s", paper.arxiv_id, exc)
       return []
   ```

5. Optional: fix the typo `"after filetering"` → `"after filtering"` in the debug log
   inside `detect_sections`.

### Verification after fix

```bash
make parse-dry
```

Expected output:

```text
--- Parsing Stats ---
   mode: dry_run
   papers_processed: 5
   papers_skipped: 0
   total_chunks: <some positive number>
   avg_chunks_per_paper: <some positive number>
   output_path: data/processed/chunks.parquet
```

---

## Step 8 — Write Tests

### 8a — Test package init

Already created in Step 2.

### 8b — `tests/test_parsing/test_chunker.py`

```python
"""Tests for the section-aware chunker.

Covers: single-section chunking, multi-section chunking, overlap correctness,
token count accuracy, unique ID generation.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.paperlens.ingestion.models import Paper
from src.paperlens.parsing.chunker import SectionAwareChunker
from src.paperlens.parsing.models import Section
from src.paperlens.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chunk_size_tokens=600,
        chunk_overlap_tokens=100,
    )


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401.00001v1",
        title="Test Paper on Attention Mechanisms",
        abstract="We propose a new attention mechanism.",
        authors=["Alice Test", "Bob Mock"],
        categories=["cs.LG"],
        published=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2024, 1, 2, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        pdf_path="/tmp/2401.00001v1.pdf",
    )


@pytest.fixture
def short_section() -> Section:
    """A section that fits in a single chunk."""
    return Section(
        label="abstract",
        start_page=1,
        end_page=1,
        text="This is a short abstract. " * 10,
        confidence=1.0,
    )


@pytest.fixture
def long_section() -> Section:
    """A section that requires multiple chunks."""
    # ~1500 tokens of text — should produce 3 chunks with 600/100 settings.
    paragraph = (
        "We propose a novel method for training large language models. "
        "Our approach combines gradient descent with adaptive learning rates. "
        "Experiments on standard benchmarks show consistent improvements. "
    )
    return Section(
        label="method",
        start_page=3,
        end_page=8,
        text=paragraph * 80,
        confidence=1.0,
    )


class TestSingleChunk:
    def test_short_section_produces_one_chunk(
        self, settings: Settings, sample_paper: Paper, short_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=short_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert len(chunks) == 1
        assert chunks[0].section_label == "abstract"
        assert chunks[0].chunk_index == 0

    def test_single_chunk_has_correct_total_counts(
        self, settings: Settings, sample_paper: Paper, short_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=short_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert chunks[0].total_chunks_in_section == 1


class TestMultiChunk:
    def test_long_section_produces_multiple_chunks(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        assert len(chunks) >= 3

    def test_chunks_have_sequential_indices(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_have_unique_ids(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_token_counts_within_limit(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        for chunk in chunks:
            assert chunk.token_count <= 600

    def test_total_chunks_in_section_updated(
        self, settings: Settings, sample_paper: Paper, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_section(
            paper=sample_paper,
            section=long_section,
            section_index=0,
            total_sections=1,
            chunk_offset=0,
        )
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_section == total


class TestChunkPaper:
    def test_multiple_sections_produce_flat_chunk_list(
        self, settings: Settings, sample_paper: Paper, short_section: Section, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [short_section, long_section])
        assert len(chunks) >= 2

    def test_total_chunks_in_paper_updated(
        self, settings: Settings, sample_paper: Paper, short_section: Section, long_section: Section
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [short_section, long_section])
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_paper == total

    def test_empty_sections_produce_no_chunks(
        self, settings: Settings, sample_paper: Paper
    ) -> None:
        chunker = SectionAwareChunker(settings)
        chunks = chunker.chunk_paper(sample_paper, [])
        assert chunks == []
```

### 8c — `tests/test_parsing/test_pdf_parser.py`

```python
"""Tests for the PDF parser.

Covers: section detection from synthetic page texts, heading matching, filtering.
"""

import pytest

from src.paperlens.parsing.pdf_parser import PdfParser
from src.paperlens.parsing.models import Section


class TestHeadingDetection:
    def test_detects_numbered_introduction(self) -> None:
        parser = PdfParser()
        pages = [
            "Title\nAuthors\nAbstract text here.",
            "1. Introduction\nWe study the problem of...",
            "2. Related Work\nPrior work has shown...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "introduction" in labels

    def test_detects_method_section(self) -> None:
        parser = PdfParser()
        pages = [
            "Abstract\nThis paper presents...",
            "3. Method\nOur approach consists of...",
            "3.1 Architecture\nThe model has...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "method" in labels

    def test_filters_references(self) -> None:
        parser = PdfParser()
        pages = [
            "1. Introduction\nWe present...",
            "5. Conclusion\nIn summary...",
            "References\n[1] Smith et al...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "references" not in labels

    def test_empty_pages_returns_empty_list(self) -> None:
        parser = PdfParser()
        assert parser.detect_sections([]) == []

    def test_pages_before_first_heading_assigned_to_abstract(self) -> None:
        parser = PdfParser()
        pages = [
            "Title and abstract text on page 1.",
            "Some more abstract content.",
            "1. Introduction\nNow we begin...",
        ]
        sections = parser.detect_sections(pages)
        assert sections[0].label == "abstract"
        assert sections[0].start_page == 1
```

Run all tests to confirm they pass before proceeding:

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: all placeholder tests + all parsing tests pass. No PDF files are needed for
these tests — they use synthetic page text.

---

## Step 9 — Update `.env.example`

The following configurable parameters are already defined in `.env.example`:

```bash
# --- Paths ---
PROCESSED_CHUNKS_PATH=./data/processed/chunks.parquet

# --- Embeddings & Retrieval ---
CHUNK_SIZE_TOKENS=600
CHUNK_OVERLAP_TOKENS=100
```

These values are read by `SectionAwareChunker` via the `Settings` class.

### Verification

- [X] `.env.example` has `CHUNK_SIZE_TOKENS` and `CHUNK_OVERLAP_TOKENS` (already present)
- [X] `.env` has matching values
- [X] `src/paperlens/settings.py` has corresponding settings fields (already present)

---

## Step 10 — Run Parsing and Record Statistics

### 10a — Dry run first (mandatory)

```bash
source .venv/bin/activate
make parse-dry
```

Confirm `papers_processed: 5`, `total_chunks > 0`, and `mode: dry_run`. If `total_chunks`
is 0, check that `data/raw/metadata.jsonl` has records with non-null `pdf_path`.

### 10b — Full parsing run

> **Time estimate:** Parsing 500 PDFs takes approximately 30–60 minutes depending on
> PDF complexity. Run in a tmux pane so it survives terminal disconnects.

```bash
tmux new -s parse
source .venv/bin/activate
make parse
# Ctrl+B D to detach; tmux attach -t parse to re-attach
```

If the run is interrupted, re-run `make parse` — already-parsed papers are skipped.

### 10c — Verify the chunk dataset and record stats

Run this after parsing completes and paste the output into `docs/chunking.md`:

```bash
source .venv/bin/activate

python - <<'EOF'
import pandas as pd

df = pd.read_parquet("data/processed/chunks.parquet")

print(f"Total chunks          : {len(df)}")
print(f"Unique papers parsed  : {df['arxiv_id'].nunique()}")
print(f"Avg chunks/paper      : {len(df) / df['arxiv_id'].nunique():.1f}")
print(f"Avg token_count       : {df['token_count'].mean():.1f}")
print(f"Median token_count    : {df['token_count'].median():.1f}")
print(f"Min token_count       : {df['token_count'].min()}")
print(f"Max token_count       : {df['token_count'].max()}")

print("\nChunks per section:")
print(df["section_label"].value_counts().to_string())

print("\nSample chunk:")
row = df.iloc[len(df) // 2]
print(f"  chunk_id    : {row['chunk_id']}")
print(f"  arxiv_id    : {row['arxiv_id']}")
print(f"  section     : {row['section_label']}")
print(f"  pages       : {row['page_start']}–{row['page_end']}")
print(f"  token_count : {row['token_count']}")
print(f"  text preview: {row['text'][:120]}...")
EOF

echo ""
echo "Disk usage:"
du -sh data/processed/chunks.parquet
```

---

## Step 11 — Documentation: `docs/chunking.md`

Create `docs/chunking.md`. Fill in the placeholders with actual numbers from Step 10c.

Key sections to include:
- Parameters table showing `.env` configuration
- Section detection strategy
- Chunking strategy with overlap justification
- Dataset statistics
- Chunk schema
- Sample chunk
- How to re-run parsing

---

## Step 12 — Commit and Push

```bash
cd ~/ML-Projects/paperlens
source .venv/bin/activate

pre-commit run --all-files
# Fix any ruff issues, then re-stage modified files

git add \
  requirements.txt \
  src/paperlens/parsing/ \
  scripts/parse.py \
  Makefile \
  tests/test_parsing/ \
  docs/chunking.md

# Confirm data files are NOT staged
git status | grep "data/processed" && echo "⚠ data/processed files staged — unstage them" || echo "✓ No data files staged"

git commit -m "$(cat <<'EOF'
Complete Milestone 1.2: document parsing and section-aware chunking.

- Add tiktoken to requirements.txt.
- Add src/paperlens/parsing: Section/Chunk models, PdfParser, SectionAwareChunker, ParsingPipeline.
- SectionAwareChunker now reads chunk_size_tokens and chunk_overlap_tokens from Settings.
- Add scripts/parse.py CLI (--dry-run, --limit, --force).
- Add Makefile targets: parse, parse-dry.
- Add tests/test_parsing: chunker and pdf_parser tests (all passing).
- Parse 500 PDFs into data/processed/chunks.parquet (~15-25k chunks).
- Document chunking strategy and schema in docs/chunking.md.
EOF
)"

git push
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: fitz` | PyMuPDF not installed | `pip install pymupdf` (already in `requirements.txt`) |
| `ModuleNotFoundError: tiktoken` | not installed | `pip install tiktoken` |
| `ModuleNotFoundError: pyarrow` | Parquet backend missing | `pip install pyarrow` (already in `requirements.txt`) |
| `papers_processed: 0` | no PDFs found | Check `data/raw/metadata.jsonl` has records with non-null `pdf_path` |
| `total_chunks: 0` | section detection failed | Run `make parse-dry` and inspect logs; check PDF text extraction |
| `Parser runs very slow` | large PDFs or many pages | Normal for first run; subsequent runs skip already-parsed papers |
| `OSError: [Errno 28] No space left` | disk full | `du -sh data/processed/`; free disk space |
| `pytest` fails on `test_chunker` | chunker expects Settings object | Tests now use `Settings` fixture |
| `ruff` reports issues in new files | formatting not applied | `make format`, then `git add` the modified files |
| `pre-commit run` modifies files before commit | normal hook behaviour | `git add -u` then retry the commit |

---

## Final Verification Gate

Run through this checklist before archiving this milestone. **All must pass.**

| # | Check | Pass? |
|---|---|---|
| 1 | `tiktoken>=0.7.0` in `requirements.txt`; importable in `.venv` | [X] |
| 2 | `from src.paperlens.parsing.models import Chunk, Section` — no error | [X] |
| 3 | `from src.paperlens.parsing.pdf_parser import PdfParser` — no error | [X] |
| 4 | `from src.paperlens.parsing.chunker import SectionAwareChunker` — no error | [X] |
| 5 | `SectionAwareChunker` reads `chunk_size_tokens` and `chunk_overlap_tokens` from `Settings` | [X] |
| 6 | `from src.paperlens.parsing.pipeline import ParsingPipeline` — no error | [X] |
| 7 | `python scripts/parse.py --help` prints usage | [X] |
| 8 | `make parse-dry` completes with `papers_processed: 5`, `total_chunks > 0` | [X] |
| 9 | `pytest tests/ -v` — all tests pass | [X] |
| 10 | `make parse` completed; `data/processed/chunks.parquet` has ≥10,000 chunks | [X] |
| 11 | `data/processed/chunks.parquet` has required columns: `chunk_id`, `arxiv_id`, `section_label`, `text`, `token_count` | [X] |
| 12 | `docs/chunking.md` exists with actual statistics filled in | [X] |
| 13 | `git push` succeeds; no `data/` files in the commit | [X] |
| 14 | CI workflow passes on GitHub | [X] |
| 15 | Blueprint Milestone 1.2 checkboxes checked in `plan/blueprint.md` | [X] |

---

## On Completion — Archive and Advance

1. **Archive** — create `plan/completed-milestones/phase-1-core-rag-pipeline/1.2-document-parsing-chunking.md` with the milestone summary and verbatim plan content, then add a summary row to `plan/completed_milestones.md`
2. **Blueprint** — check off Milestone 1.2 items in `plan/blueprint.md`
3. **Replace this file** with **Milestone 1.3 — Embeddings & Vector Store**
4. **GitHub Projects** — move 1.2 card to Done

---

## Next Milestone Preview

**Milestone 1.3 — Embeddings & Vector Store**

Embed all chunks from `data/processed/chunks.parquet` using `BAAI/bge-large-en-v1.5` via
sentence-transformers. Store embeddings + metadata in ChromaDB. Implement basic semantic
search (top-k retrieval). Add index rebuild script for reproducibility. Document embedding
model choice, index size, and retrieval latency baseline in `docs/retrieval-baseline.md`.

See `plan/blueprint.md` → Milestone 1.3.
