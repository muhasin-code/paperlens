"""
ArxivFetcher: search arXiv, download PDFs, and store metadata.

Design decisions:
- All outputs (metadata.jsonl, PDFs, log) lives under data/raw/ (gitignored).
- Idempotent: existing arxiv_ids are loaded on startup and skipped on re-run.
- Sequential downloades with configurable delay to respect arXiv rate limits.
- Dual logging: console (INFO) + file (DEBUG) for full auditability.
"""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import arxiv
from tqdm import tqdm

from src.paperlens.ingestion.models import Paper
from src.paperlens.settings import Settings


def _setup_logger(log_path: Path) -> logging.Logger:
    """Return a logger writing to console (INFO) and a file (DEBUG)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("paperlens.ingestion")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import in tests

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class ArxivFetcher:
    """
    Fetches arXiv paper metadata and downloads PDF to local disk.

    Args:
        settings: Application settings controlling category, dates, and paths.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = _setup_logger(settings.ingestion_log)
        self.client = arxiv.Client(
            page_size=100, delay_seconds=settings.arxiv_delay_seconds, num_retries=3
        )

    # ── Query ─────────────────────────────────────────────────────────────
    def _build_query(self) -> str:
        """Build the arXiv search query with category and date range filters."""
        return f"cat: {self.settings.arxiv_category}"

    # ── Metadata storage ──────────────────────────────────────────────────
    def load_existing_ids(self) -> set[str]:
        """Return the set of arxiv_ids already present in metadata.jsonl."""
        path = self.settings.metadata_path
        if not path.exists():
            return set()

        ids: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = Paper.from_jsonl(line)
                    ids.add(paper.arxiv_id)
                except Exception as exc:
                    self.logger.warning("Could not parse metadata line: %s", exc)

        self.logger.info("Loaded %d existing paper IDs from metadata.", len(ids))
        return ids

    def _append_paper(self, paper: Paper) -> None:
        """Append one Paper record to metadata.jsonl (creates file if absent)."""
        path = self.settings.metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(paper.to_jsonl())

    # ── Fetch ─────────────────────────────────────────────────────────────
    def fetch_metadata(self, skip_ids: set[str]) -> list[Paper]:
        """
        Fetch paper metadata from arXiv without downloading PDFs.

        Args:
            skip_ids: Set of arxiv_ids to skip (already registered).

        Returns:
            List of Paper objects for new papers only.
        """
        query = self._build_query()
        self.logger.info("Query: %s | max_results=%d", query, self.settings.arxiv_max_results)

        search = arxiv.Search(
            query=query,
            max_results=self.settings.arxiv_max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        date_from = datetime.strptime(self.settings.arxiv_date_from, "%Y-%m-%d").replace(tzinfo=UTC)
        date_to = datetime.strptime(self.settings.arxiv_date_to, "%Y-%m-%d").replace(tzinfo=UTC)

        papers: list[Paper] = []
        skipped = 0

        for result in tqdm(
            self.client.results(search),
            total=self.settings.arxiv_max_results,
            desc="Fetching metadata",
            unit="paper",
        ):
            if not (date_from <= result.published <= date_to):
                continue

            arxiv_id = result.get_short_id()

            if arxiv_id in skip_ids:
                skipped += 1
                self.logger.debug("Skip (exists): %s", arxiv_id)
                continue

            paper = Paper(
                arxiv_id=arxiv_id,
                title=result.title.strip(),
                abstract=result.summary.strip(),
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published=result.published.astimezone(UTC),
                updated=result.updated.astimezone(UTC),
                pdf_url=result.pdf_url or "",
            )
            papers.append(paper)
            self.logger.debug("Queued: %s - %s", arxiv_id, paper.title[:60])

        self.logger.info(
            "Metadata fetch done. New %d | Skipped (already registered): %d", len(papers), skipped
        )
        return papers

    # ── Download ──────────────────────────────────────────────────────────
    def download_pdfs(self, papers: list[Paper]) -> list[Paper]:
        """
        Download PDFs for each paper. Update pdf_path on success.

        Papers whose pdf_url is empty or whose download fails are logged and retained with pdf_path=None. They will be excluded in Milestone 1.2.

        Returns:
            Updated list of Paper objects with pdf_path set where available.
        """
        pdf_dir = self.settings.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)

        updated: list[Paper] = []
        failed = 0

        for paper in tqdm(papers, desc="Downloading PDFs", unit="pdf"):
            dest = pdf_dir / f"{paper.arxiv_id}.pdf"

            # Already on disk (e.g., script interrupted and restarted)
            if dest.exists():
                self.logger.debug("Already on disk: %s", dest.name)
                updated.append(paper.model_copy(update={"pdf_path": str(dest.resolve())}))
                continue

            if not paper.pdf_url:
                self.logger.warning("No PDF URL for %s - skipping.", paper.arxiv_id)
                updated.append(paper.model_copy(update={"pdf_pat": None}))
                failed += 1
                continue

            try:
                result = next(self.client.results(arxiv.Search(id_list=[paper.arxiv_id])))
                result.download_pdf(
                    dirpath=str(pdf_dir),
                    filename=f"{paper.arxiv_id}.pdf",
                )
                updated.append(paper.model_copy(update={"pdf_path": str(dest.resolve())}))
                self.logger.debug("Downloaded: %s", dest.name)
            except Exception as exc:
                self.logger.warning("Download failed for %s: %s", paper.arxiv_id, exc)
                updated.append(paper.model_copy(update={"pdf_path": None}))
                failed += 1

            time.sleep(self.settings.arxiv_delay_seconds)

        self.logger.info(
            "PDF download done. Success: %d | Failed/Skipped: %d", len(updated) - failed, failed
        )
        return updated

    # ── Full Pipeline ─────────────────────────────────────────────────────
    def run(self, dry_run: bool = False) -> dict:
        """
        Execute the full ingestion pipeline.

        Args:
            dry_run: If True, fetch and log metadata only - nothing written to disk.

        Returns:
            Dict of stats: mode, existing_papers, new_papers, pdfs_downloaded, pdfs_failed, total_corpus.
        """
        self.logger.info("=== PaperLens Ingestion Start | dry_run=%s ===", dry_run)

        existing_ids = self.load_existing_ids()
        new_papers = self.fetch_metadata(skip_ids=existing_ids)

        if dry_run:
            self.logger.info(
                "[DRY RUN] would process %d new papers. Nothing written.", len(new_papers)
            )
            return {
                "mode": "dry_run",
                "existing_papers": len(existing_ids),
                "new_papers_found": len(new_papers),
                "pdfs_downloaded": 0,
            }

        papers_with_pdfs = self.download_pdfs(new_papers)

        for paper in papers_with_pdfs:
            self._append_paper(paper=paper)

        downloaded = sum(1 for p in papers_with_pdfs if p.pdf_path is not None)
        failed = len(papers_with_pdfs) - downloaded

        self.logger.info(
            "=== Ingestion Complete | total=%d | new=%d | downloaded=%d | failed=%d ===",
            len(existing_ids) + len(papers_with_pdfs),
            len(papers_with_pdfs),
            downloaded,
            failed,
        )

        return {
            "mode": "full",
            "existing_papers": len(existing_ids),
            "new_papers": len(papers_with_pdfs),
            "pdfs_downloaded": downloaded,
            "pdfs_failed": failed,
            "total_corpus": len(existing_ids) + len(papers_with_pdfs),
            "metadata_path": str(self.settings.metadata_path),
            "pdf_dir": str(self.settings.pdf_dir),
        }
