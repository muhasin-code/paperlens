# PaperLens — Data Ingestion

## Domain Choice

**Primary category:** cs.LG (Machine Learning)

**Date range:** 2023-01-01 to 2026-12-31

### Rationale

cs.LG was selected as the corpus domain for three reasons:

1. **Interview relevance.** cs.LG covers the ML engineering topics most likely to surface
   in technical interviews — architectures, optimisation, generalisation, and evaluation
   methodology. A system that can synthesise across these papers has an immediately
   credible use case.

2. **Cross-paper citation density.** Papers in cs.LG cite each other heavily, making
   cross-paper synthesis queries substantive rather than trivial. This is important for
   Phase 2 retrieval benchmarking, where precision@5 differences need to be meaningful.

3. **Corpus coherence.** A single category produces a thematically coherent corpus.
   Mixing categories in Phase 1 would complicate the evaluation story in Phase 2–4.

**Alternatives considered:**
- cs.CL (NLP) — viable, narrower focus on language models specifically.
- cs.AI (AI general) — broader but thinner per-topic depth.
Both can be added as secondary ingestion runs once the pipeline is validated.

---
Total papers       : 500
Date range         : 2026-06-19 → 2026-06-23
PDFs downloaded    : 499/500 (99%)
Top categories:
  cs.LG: 500
  cs.AI: 174
  cs.CL: 52
  cs.CV: 51
  stat.ML: 42

Disk usage:
2.2G	data/raw/pdfs/
PDF count:
499

## Dataset Statistics

| Metric | Value |
|---|---|
| Total papers ingested | 500 |
| PDFs downloaded | 499 |
| PDF coverage | 99% |
| Date range (actual) | 2026-06-19 -> 20206-06-23 |
| Total PDF disk usage | 2.2 GB |
| Primary category | cs.LG |
| Top secondary categories | cs.AI, cs.CL, cs.CV |
| Ingestion date | 2026-06-24 |
| Ingestion duration | 1 Hour |

---

## How to Re-Run Ingestion

The fetcher is idempotent. Re-running skips papers already in `metadata.jsonl` and only downloads new ones.

To extend the corpus with the same settings:

```bash
source .venv/bin/activate
make ingest
```

To preview what would be added without writing anything:

```bash
make ingest-dry
```

To ingest a different category or date range without editing `.env`:

```bash
python scripts/ingest.py --category cs.CL --max-results 200
```

---

## Data Layout

```
data/raw/                          # gitignored — local only
  metadata.jsonl                   # ~<N> records; ~<size on disk>
  pdfs/                            # ~<N> PDF files; ~<total size>
    2301.07597v2.pdf
    2302.14838v1.pdf
    ...
  ingestion.log                    # full ingestion trace with timestamps
```

To share the dataset, export `metadata.jsonl` and `pdfs/` to a shared volume or a HuggingFace Dataset repository (documented in Phase 6).

---

## Metadata Schema

Each line in `metadata.jsonl` is a JSON object matching the `Paper` Pydantic model
(`src/paperlens/ingestion/models.py`):

| Field | Type | Description |
|---|---|---|
| `arxiv_id` | str | Short ID with version, e.g. `2301.07597v2` |
| `title` | str | Paper title |
| `abstract` | str | Full abstract text |
| `authors` | list[str] | Author names in order |
| `categories` | list[str] | arXiv category tags |
| `published` | datetime (ISO 8601, UTC) | Original submission date |
| `updated` | datetime (ISO 8601, UTC) | Last revision date |
| `pdf_url` | str | Canonical arXiv PDF URL |
| `pdf_path` | str or null | Absolute local path to downloaded PDF |
| `ingested_at` | datetime (ISO 8601, UTC) | When this record was written |

---

## Known Limitations

- **Sequential downloads.** PDFs are fetched one at a time to respect arXiv's rate-limit
  policy. Concurrent downloading is not implemented in Phase 1.
- **Abstracts only in metadata.** Full-text extraction from PDFs happens in Milestone 1.2
  (PyMuPDF parsing and chunking).
- **PDF failures.** A small number of papers (~1–3%) may not have a publicly accessible
  PDF. These are stored with `pdf_path: null` and will be excluded in Milestone 1.2.
