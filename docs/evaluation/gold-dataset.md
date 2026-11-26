# Gold evaluation dataset

`eval/gold_dataset.jsonl` holds both the retrieval labels and generation
references used to evaluate PaperLens's conversational RAG path:

- **Retrieval:** 26 rows marked with `retrieval` in `eval_sets`, with 10
  relevant chunk IDs each. Retrieval metrics are documented in
  [retrieval-metrics.md](./retrieval-metrics.md).
- **Generation:** 75 rows marked with `generation` in `eval_sets`. The
  original 26 retrieval rows are also generation rows; the other 49 rows are
  generation-only.

The frozen pre-milestone retrieval labels are in
`eval/gold_dataset.v2.5-retrieval.jsonl`. Scripts read the live
`eval/gold_dataset.jsonl`, not the frozen copy. Extraction gold
(`eval/gold_extraction.jsonl`) is a separate dataset and is not part of this
evaluation.

## Schema

| Field | Required | Rule |
|---|---|---|
| `id` | Yes | Unique ID in file order (`qa_001` through `qa_075`). |
| `dataset_version` | Yes | `4.3.0`. |
| `query` | Yes | Non-empty and unique. |
| `ground_truth` | Generation rows | Answer reference supported by its source chunks; no chunk-ID citation markers. |
| `relevant_chunk_ids` | Yes | Non-empty retrieval labels. The original 26 rows retain their 10 labels; generation-only rows use their 1-3 source chunks. |
| `relevant_arxiv_ids` | Yes | Non-empty paper-level retrieval labels. |
| `source_chunk_ids` | Generation rows | 1-3 valid corpus chunk IDs supporting `ground_truth`. On original retrieval rows, these can differ from `relevant_chunk_ids` so frozen retrieval labels remain unchanged. |
| `eval_sets` | Yes | Original 26 rows: `["retrieval", "generation"]`; added rows: `["generation"]`. |
| `difficulty` | No | `easy`, `medium`, or `hard`. |
| `category` | No | `concept`, `comparison`, `architecture`, `training`, or `generation`. |
| `notes` | No | Annotation notes; original retrieval-row notes are preserved. |

Ragas uses `query` as the user input, the generated answer as the response,
retrieved chunk text as the contexts, and `ground_truth` as the reference.

## Annotation rules

Write `ground_truth` only from the text in `data/processed/chunks.parquet`
identified by `source_chunk_ids`. Do not use outside knowledge, paper titles
alone, or short previews from discovery output. If the selected chunks do not
support an answer, choose different source chunks.

Generation-only rows must not be added to the retrieval benchmark. A short
source list is not a substitute for the retrieval benchmark's relevance
labels. `scripts/evaluate_retrieval.py` scores rows marked `retrieval`;
Ragas context precision scores the chunks actually passed to the generator.
These are different metrics and should not be compared as if they were the
same measurement.

When adding a row, mark it `["generation"]` unless it has also been given
retrieval labels using the retrieval benchmark's annotation process. Then run:

```bash
make validate-gold-schema
make validate-gold
```

## Run Ragas locally

The dry run checks the dataset without loading a model or making network
requests:

```bash
make eval-ragas-dry
```

To reproduce the recorded run, use `--top-k 5`:

```bash
LANGFUSE_ENABLED=false python scripts/evaluate_ragas.py \
  --top-k 5 \
  --output eval/ragas_baseline.json
```

The run uses the configured generation and judge models and the local BGE
embedding model. `--save-answers` can persist generated answers so scoring can
be repeated without regenerating them. `--resume` can continue from answer and
score checkpoints if a run is interrupted. Keep local answer and checkpoint
files out of commits.

The reported `retrieval_top_k` is 5. This run does not enforce the
faithfulness threshold; CI gating is a later milestone.

## Baseline

This table records one local run. Scores may differ on later runs because of
model and provider variation.

| Metric | Mean | n | n_skipped_empty_context | llm_model | judge_model | ragas_version | retrieval_top_k | date |
|---|---:|---:|---:|---|---|---|---:|---|
| faithfulness | NaN* | 75 | 0 | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | 0.4.3 | 5 | 2026-11-25 |
| answer_relevancy | 0.7807 | 75 | 0 | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | 0.4.3 | 5 | 2026-11-25 |
| context_precision | 0.8353 | 75 | 0 | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | 0.4.3 | 5 | 2026-11-25 |

The source report is [`eval/ragas_baseline.json`](../../eval/ragas_baseline.json).
It records 75 scored rows and does not enforce a threshold. `NaN*` means the
faithfulness result is not a usable aggregate: five row values are NaN
(`qa_006`, `qa_011`, `qa_023`, `qa_024`, and `qa_028`), so only 70 values are
finite. The arithmetic mean of those 70 finite values is 0.9736; it is
provided for context and is not the 75-row mean reported in the JSON.

Milestone 4.4 may add a CI gate against
`settings.ragas_faithfulness_threshold` (0.75). No such gate is active in this
milestone.
