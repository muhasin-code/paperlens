# Extraction Dataset Documentation

This document describes the supervised extraction dataset prepared in Milestone 3.1 for fine-tuning Qwen2.5-3B-Instruct with QLoRA.

---

## Dataset Schema

### Output Fields

Each annotation produces a structured JSON object with the following nullable fields:

| Field            | Type              | Description                                                               |
|------------------|-------------------|-----------------------------------------------------------------------------|
| `research_question`| `string|null`   | The core problem or hypothesis the paper addresses                        |
| `method`         | `string|null`   | The proposed approach or algorithm (one paragraph max)                     |
| `datasets`       | `string[]|null`  | List of dataset names used                                                 |
| `metrics`        | `string[]|null`  | List of evaluation metrics reported                                      |
| `key_finding`    | `string|null`   | The main result or contribution in one sentence                            |

### Input Format

Each training example consists of:
- **input**: The full text of a paper chunk from `data/processed/chunks.parquet`
- **output**: The structured JSON object extracted from that chunk

### TRL SFT Format

The dataset is formatted for HuggingFace TRL's SFTTrainer with a chat template:

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "Extract structured information from the following ML paper chunk:\n\n<chunk text>"},
    {"role": "assistant", "content": "{\"research_question\": \"...\", \"method\": \"...\", \"datasets\": [...], \"metrics\": [...], \"key_finding\": \"...\"}"}
  ],
  "metadata": {
    "chunk_id": "2606.22406v2_chunk_0007",
    "arxiv_id": "2606.22406v2",
    "section_label": "method"
  }
}
```

---

## Annotation System Prompt

The default system prompt instructs the teacher LLM to extract structured information:

```
You are a research assistant specialized in extracting structured information from machine learning papers.

From the provided paper chunk, extract the following fields and return ONLY a JSON object with these keys (all nullable):

- research_question: The core problem or hypothesis the paper addresses
- method: The proposed approach or algorithm (max one paragraph)
- datasets: List of dataset names used (empty array if none)
- metrics: List of evaluation metrics reported (empty array if none)
- key_finding: The main result or contribution in one sentence

If a field is not present or not useful, set it to null. Do not hallucinate information. Do not add any prose - only return the JSON object.

Example output:
{"research_question": "How to improve...", "method": "...", "datasets": ["..."], "metrics": ["..."], "key_finding": "..."}
```

The user turn is:

```
Extract structured information from the following ML paper chunk:

<chunk text>
```

---

## Section Filter Rationale

Chunks are filtered to only those from the following sections:

- **method**: Contains methodological details, algorithm descriptions, architectural choices
- **experiments**: Contains experimental setup, hyperparameters, training procedures
- **results**: Contains quantitative results, figures, main findings
- **evaluation**: Contains evaluation metrics, comparison tables, analysis

### Excluded Sections and Why

| Section      | Reason for Exclusion                                                                                  |
|--------------|-------------------------------------------------------------------------------------------------------|
| `abstract`   | Typically summarizes; rarely contains method-level detail at chunk granularity; often overlapping with other sections |
| `introduction` | Contains background and motivation, not extractable structured data                                 |
| `related_work` | Describes OTHER papers, not the target paper's method or findings                                    |
| `references` | No extractable structured information; just citations                                                |
| `acknowledgements` | No method/experiment/result content                                                                |
| `appendix` | Supplementary details; often contains raw data, proofs, or extended tables not suitable for extraction |
| `discussion` | Contains interpretation and limitations; less structured than results/evaluation                     |
| `conclusion` | Summary statements; rarely contains novel methodological content                                     |
| `background` | Foundational theory; not actionable extraction                                                     |

Only method, experiments, results, and evaluation sections reliably contain content that maps to the five extraction fields.

---

## Train/Val Split Methodology

The dataset is split 90/10 into training and validation sets:

- **Split ratio**: 90% train, 10% validation (configurable via `EXTRACTION_VAL_SPLIT=0.1`)
- **Method**: Random shuffle with fixed seed for reproducibility, then split at ratio boundary
- **No overlap**: Each chunk_id appears in exactly one split

The small validation set (10%) is sufficient for QLoRA fine-tuning evaluation, as the model is fine-tuned on the entire training set and early stopping can be applied based on validation loss.

---

## Dataset Statistics

Files are written to `data/extraction_dataset/`:

| File                    | Purpose                                        |
|-------------------------|------------------------------------------------|
| `train.jsonl`           | Training examples (90% of total)               |
| `val.jsonl`             | Validation examples (10% of total)             |
| `checkpoint.jsonl`      | Idempotency checkpoint for resumable annotation  |

### Configuration

| Setting                    | Default         | Description                                    |
|----------------------------|-----------------|------------------------------------------------|
| `EXTRACTION_MAX_EXAMPLES`  | 2000            | Maximum training examples to generate           |
| `EXTRACTION_SECTIONS`      | method,experiments,results,evaluation | Sections to include |
| `EXTRACTION_VAL_SPLIT`     | 0.1             | Fraction of data for validation                |
| `EXTRACTION_CHECKPOINT_PATH`| ./data/extraction_dataset/checkpoint.jsonl | Checkpoint file path |

---

## Generation Command

```bash
# Full run (generates up to 2000 examples)
make build-extraction-dataset

# Dry run (5 examples, no writes)
make build-extraction-dry

# Custom limit and sections
python scripts/build_extraction_dataset.py --limit 500 --sections method,results

# Force re-annotation
python scripts/build_extraction_dataset.py --force

# Validate generated dataset
make validate-extraction-dataset
```

---

## Teacher Model

The annotation uses the configured LLM provider:
- **Default (Phase 3 setup)**: Groq `llama-3.3-70b-versatile` via OpenAI-compatible API
- **Fallback**: Ollama `phi4-mini` or other local models

All API calls go through `src/paperlens/llm/` using `get_llm_provider(settings)`. Rate limits on Groq free tier (30 RPM, 6,000 TPM) are handled with exponential backoff in the annotator.

---

## Source Files

| File                                                | Purpose                                                    |
|-----------------------------------------------------|------------------------------------------------------------|
| `src/paperlens/extraction/models.py`                | `ExtractionExample`, `ExtractionOutput` Pydantic models    |
| `src/paperlens/extraction/annotator.py`             | `ChunkAnnotator` with LLM annotation and validation        |
| `src/paperlens/extraction/pipeline.py`              | `ExtractionPipeline` for full dataset generation           |
| `scripts/build_extraction_dataset.py`               | CLI entry point                                            |
| `scripts/validate_extraction_dataset.py`            | Dataset validation script                                  |
| `src/paperlens/settings.py`                         | Extraction config fields                                   |
| `.env.example`                                      | Extraction env variable placeholders                        |

---

## Troubleshooting

| Problem                                           | Cause                                                                        | Fix                                                               |
|---------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `chunks.parquet not found`                       | Ingestion/parsing not complete                                               | Run `make parse` first                                             |
| `Rate limit error` from Groq                     | Free tier rate limit exceeded                                                | Pipeline waits and retries with exponential backoff               |
| `AnnotationError: not valid JSON`                | LLM output deviated from expected JSON format                                | Check system prompt; may need prompt tuning                        |
| `AnnotationError: does not match extraction schema` | LLM returned JSON with missing fields                                      | Review LLM output; fields are all nullable, should not happen      |
| `train.jsonl is empty`                           | No chunks in filter set or all failed annotation                           | Check `extraction_sections` config; check logs for errors          |
| `Chunk ID in both train and val`                 | Validation bug                                                               | Re-run validation; report if persists                              |
| Val ratio not 10%                                | Small dataset rounding                                                       | Manual check with `validate_extraction_dataset.py`                 |
| API key error                                    | `LLM_API_KEY` not set or invalid                                             | Set `LLM_API_KEY` in `.env` or `.env.local`                        |
| Timeout on generate                                | Network latency or model loading                                             | Increase timeout in `OpenAICompatProvider` if needed             |

---

*Document created: 2026-09-17*
