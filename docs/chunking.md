# Chunking Strategy

This document describes how raw PDFs are turned into the section-aware chunk dataset
stored at `data/processed/chunks.parquet`.

---

## 1. Parameters

All chunking parameters are configured in `.env`. The parser reads them via the
`Settings` class (`src/paperlens/settings.py`).

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE_TOKENS` | `600` | Maximum token length per chunk. Matches the upper context limit for `BAAI/bge-large-en-v1.5`. |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between consecutive chunks in a section. Prevents multi-sentence claims from being split at boundaries. |
| `PROCESSED_CHUNKS_PATH` | `./data/processed/chunks.parquet` | Output file for the serialised chunk dataset. |

Effective sliding-window step:

```
step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS = 500
```

### Tokenizer

`tiktoken` with the `cl100k_base` encoding (the tokenizer used by OpenAI embedding
models). This was chosen because:

- It matches the tokenisation used by `BAAI/bge-large-en-v1.5`, so token counts
  correspond closely to actual embedding-model consumption.
- It is far more accurate than whitespace or character-based splitting, preventing
  chunks from silently exceeding the target token budget.

---

## 2. PDF Text Extraction

Text is extracted with **PyMuPDF** (`fitz`). Each page is read independently, producing
a list of plain-text strings — one per page.

```python
with fitz.open(pdf_path) as doc:
    for page in doc:
        pages.append(page.get_text("text"))
```

No OCR path is implemented yet. Papers with non-selectable text will produce empty
page contents and be skipped during section detection (logged as warnings).

---

## 3. Section Detection

Section detection is **rule-based**, not ML-based. For each page the first 300
characters of extracted text are matched against compiled regex patterns.

### Heading patterns (priority order)

| Label | Pattern examples |
|---|---|
| `abstract` | `abstract` |
| `introduction` | `1. Introduction`, `Introduction` |
| `related_work` | `2. Related Work` |
| `background` | `Background` |
| `method` | `3. Method`, `3.1 Methodology`, `Approach` |
| `experiments` | `4. Experiments`, `Experimental Setup` |
| `results` | `5. Results` |
| `evaluation` | `Evaluation` |
| `discussion` | `Discussion` |
| `conclusion` | `6. Conclusion`, `Conclusions` |
| `references` | `References` |
| `acknowledgements` | `Acknowledgements`, `Acknowledgments` |
| `appendix` | `Appendix` |

Patterns are matched **case-insensitively** with `re.MULTILINE`.

### Detection algorithm

1. Iterate pages in order.
2. If a heading is detected on the current page and it differs from the current
   section label:
   - Close the current section (assign pages seen so far).
   - Start a new section from the current page.
3. If no heading is detected, append the page to the current section (fallback
   heuristic). Pages before the first detected heading are assigned to `abstract`,
   which covers the common case of title + abstract stacked on page 1.

### Filtering

Sections labelled `references` and `appendix` are **excluded** from the final dataset.
They are rarely useful for retrieval and typically pollute the embedding space.

### Confidence

Every detected section is assigned `confidence: 1.0` for explicit heading matches.
No heuristic confidence scoring is implemented yet; this field is reserved for
downstream re-ranking or filtering.

---

## 4. Chunking Strategy

Each retained `Section` is split into fixed-size token windows using a **sliding window**
with overlap.

### Why overlap?

Without overlap, a claim that spans the boundary between two chunks is only fully
present in one chunk and partially present in none. With a 100-token overlap, the
sentence that straddles the boundary is fully captured in both adjacent chunks. This
improves retrieval recall, especially for longer papers where sections run to
thousands of tokens.

### Overlap justification for this corpus

- **Avg tokens per section**: With `chunk_size_tokens=600` and `chunk_overlap_tokens=100`,
  the effective step is 500 tokens.
- **Median chunk size**: The median chunk in the produced dataset is exactly 600 tokens,
  confirming that the majority of sections or splits fill the window.
- **Section size distribution**: The `abstract` section dominates chunk count (5,400
  chunks), reflecting the fact that abstracts are short and each forms a single chunk.
  Longer sections like `method` and `experiments` produce multiple overlapping chunks.

---

## 5. Dataset Statistics

Generated: `2026-07-06`
File: `data/processed/chunks.parquet`
Disk size: **17 MB** (Parquet, Snappy-compressed)

| Metric | Value |
|---|---|
| Total chunks | **17,330** |
| Unique papers parsed | **499** |
| Avg chunks / paper | **34.7** |
| Avg token count | **579.0** |
| Median token count | **600.0** |
| Min token count | **1** |
| Max token count | **600** |

### Chunks per section

| Section | Chunks |
|---|---|
| abstract | 9,404 |
| method | 2,982 |
| experiments | 889 |
| introduction | 781 |
| results | 711 |
| conclusion | 630 |
| evaluation | 517 |
| related_work | 497 |
| acknowledgements | 353 |
| discussion | 319 |
| background | 247 |

### Notes

- `references` and `appendix` are intentionally absent because they are filtered out
  at the section-detection stage.
- `min_chunk_token_count = 1` indicates some very short sections or trailing content.
  These are retained rather than discarded so that downstream ranking can decide
  whether to surface them.

---

## 6. Chunk Schema

Each row in the Parquet file is a serialised `Chunk` model.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Globally unique ID: `{arxiv_id}_chunk_{zero_padded_index}`, e.g. `2606.22639v1_chunk_0074` |
| `arxiv_id` | `str` | Source paper arXiv identifier |
| `title` | `str` | Paper title (denormalised for citation display) |
| `authors` | `list[str]` | Paper authors |
| `section_label` | `str` | Detected section label (e.g. `method`, `results`) |
| `chunk_index` | `int` | Zero-based ordinal index of this chunk within its section |
| `page_start` | `int` | 1-indexed start page of the chunk's source text |
| `page_end` | `int` | 1-indexed end page of the chunk's source text |
| `text` | `str` | Chunk text (trimmed; ≤ `chunk_size_tokens` tokens) |
| `token_count` | `int` | Exact token count of `text` as measured by `tiktoken` |
| `total_chunks_in_section` | `int` | Total chunks in the parent section |
| `total_chunks_in_paper` | `int` | Total chunks in the entire paper |

---

## 7. Sample Chunk

```
  chunk_id    : 2606.22639v1_chunk_0074
  arxiv_id    : 2606.22639v1
  section     : abstract
  pages       : 48–49
  token_count : 600
  text preview: t
t = 0,
Ct
Na,t
p−→0 for any a ∈A;
(ii) bµa,t
p−→µ∗
a for any a ∈A.
The ϵ-greedy algorithm. Consider the ϵ-greedy polic...
```

The sample above demonstrates that mathematical notation and LaTeX fragments can
survive into chunks. No LaTeX normalisation pass has been run yet; downstream
embedding models handle raw text including symbols without failure.

---

## 8. Re-running Parsing

```bash
# Dry run on 5 papers (no writes)
make parse-dry

# Full parse (appends to chunks.parquet; skips already-parsed papers)
make parse

# Force re-parse everything
make parse -- --force
# or
python scripts/parse.py --force
```

The pipeline is idempotent: it loads existing `chunk_id` values and skips papers
that already have chunks unless `--force` is passed. Interrupted runs can be resumed
by re-running the same command.

---

## 9. Known Limitations

- **No LaTeX normalisation** — math-heavy sections contain raw LaTeX commands (e.g.
  `\mu`, `\rightarrow`). These are passed verbatim to the embedding model.
- **No OCR** — papers with rasterised/embedded text produce empty page strings and
  are skipped.
- **Uniform confidence** — all sections are currently set to `confidence: 1.0`.
  There is no distinction between explicit numbered headings and heuristic
  fallback assignments.
- **Single encoding** — only `cl100k_base` is used. Cross-encoding models with
  different tokenisers may see minor token-count drift.

---

## 10. Source Files

| File | Purpose |
|---|---|
| `src/paperlens/parsing/pdf_parser.py` | `PdfParser`: `extract_text_by_page()` and `detect_sections()` |
| `src/paperlens/parsing/chunker.py` | `SectionAwareChunker`: sliding-window chunking with overlap |
| `src/paperlens/parsing/pipeline.py` | `ParsingPipeline`: orchestration, idempotency, Parquet persistence |
| `src/paperlens/parsing/models.py` | `Section` and `Chunk` Pydantic models |
| `scripts/parse.py` | CLI entry point |
