# Citation Enforcement & Structured Responses

This document describes the citation enforcement mechanism implemented in Milestone 2.4, which ensures all RAG responses include proper citations or refuse gracefully.

---

## Response Schema

### QueryResponse

The structured response returned by `POST /query` is validated using Pydantic.


| Field                | Type              | Description                                                                             |
| -------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `answer`             | `string`          | Generated answer text with inline `[chunk_id]` citations, or refusal message            |
| `citations`          | `array[Citation]` | Source chunk references. Empty if answer is a refusal                                   |
| `confidence`         | `number`          | Confidence score (0.0 for refusal, 0.0–1.0 for valid answers based on retrieval scores) |
| `retrieval_time_ms`  | `number`          | Time spent in retrieval (embedding + search)                                            |
| `generation_time_ms` | `number`          | Time spent in LLM generation                                                            |
| `total_time_ms`      | `number`          | End-to-end latency                                                                      |




### Citation Object


| Field           | Type            | Description                                                        |
| --------------- | --------------- | ------------------------------------------------------------------ |
| `chunk_id`      | `string`        | Globally unique chunk identifier (e.g., `2401.00001v1_chunk_0001`) |
| `arxiv_id`      | `string`        | Source paper arXiv ID with version                                 |
| `title`         | `string`        | Source paper title                                                 |
| `authors`       | `array[string]` | Source paper author list                                           |
| `section_label` | `string`        | Section this chunk belongs to                                      |
| `page_start`    | `integer`       | 1-indexed starting page                                            |
| `page_end`      | `integer`       | 1-indexed ending page                                              |
| `score`         | `number`        | Retrieval similarity score                                         |
| `rank`          | `integer`       | 1-based rank in retrieved list                                     |


---



## Refusal Behavior



### When Refusal Occurs

The system returns a refusal response when:

1. **Exact match with refusal_message**: The LLM's output exactly matches the `refusal_message` from the prompt YAML, OR
2. **No citation markers**: The LLM response contains no `[chunk_id]` pattern



### Refusal Response

```json
{
  "answer": "I cannot answer from the provided sources.",
  "citations": [],
  "confidence": 0.0,
  "retrieval_time_ms": 150.2,
  "generation_time_ms": 2500.0,
  "total_time_ms": 2650.2
}
```

**Important**: The LLM output is not modified. If the model returns another text, that text is returned as-is. Only when the model explicitly refuses does the system pass through the refusal message unchanged.

---



## Prompt YAML Format



### File Location

Prompts are stored at: `{prompts_dir}/{prompt_version}.yaml`

Default path: `configs/prompts/v1.yaml`

### Schema

```yaml
---
version: v1
system_prompt: |
  You are a research assistant answering questions about ML/AI papers from arXiv.
  You are grounded --- your answers must be based ONLY on the provided context chunks.
  Cite every claim by including the [chunk_id] in square brackets at the end of sentences.
  Be concise and factual.

user_template: |
  Context chunks:
  {context}

  Question: {query}

  Answer:

refusal_message: "I cannot answer from the provided sources."
```



### Fields


| Field             | Type     | Required | Description                                                       |
| ----------------- | -------- | -------- | ----------------------------------------------------------------- |
| `version`         | `string` | Yes      | Prompt version identifier (e.g., `v1`, `v2`)                      |
| `system_prompt`   | `string` | Yes      | System-level instructions for the LLM                             |
| `user_template`   | `string` | Yes      | User message template with `{context}` and `{query}` placeholders |
| `refusal_message` | `string` | Yes      | Exact text the LLM should return when evidence is insufficient    |




### Placeholder Substitution

The `{context}` placeholder is replaced with:

```
[{chunk_id}] {chunk_text}
---
[{chunk_id}] {chunk_text}
```

The `{query}` placeholder is replaced with the user's natural-language question.

---



## Creating a New Prompt Version

To create a new prompt version without code changes:

1. **Create a new YAML file** in `configs/prompts/` (e.g., `v2.yaml`)
2. **Update version identifier**:

```yaml
version: v2
```

1. **Test the new prompt locally**:

```bash
source .venv/bin/activate
python -c "
from src.paperlens.settings import Settings
from src.paperlens.api.prompt_loader import PromptLoader
```



# Load new version

settings = Settings(prompt_version='v2')
loader = PromptLoader(settings)
prompt = loader.load_prompt()
print('Loaded v2:', prompt.version)

# Build test prompt

test_prompt = prompt.build_prompt(context='[chunk_001] sample text', query='test?')
print('Prompt built successfully')
"

```

4. **Switch by updating PROMPT_VERSION** in `.env` or passing explicitly:

```python
loader = PromptLoader(settings)
prompt = loader.load_prompt('v2')  # Override version
```

1. **Commit the new YAML file**:

```bash
git add configs/prompts/v2.yaml
git commit -m "Add v2 prompt with refined citation instructions"
```

---

## Retry Logic for Malformed Outputs

### When Retry Happens

After LLM generation, the response is validated:

- **Valid**: Response is non-empty and contains at least one `[chunk_id]` pattern
- **Malformed**: Response is empty or lacks citation markers

### Retry Behavior


| Attempt | Model Used              | Condition                  |
| ------- | ----------------------- | -------------------------- |
| 1       | Primary (`llama3.2:1b`) | First generation           |
| 2       | Primary (`llama3.2:1b`) | Retry for malformed output |
| 3       | Fallback (`phi4-mini`)  | If 2 retries exhausted     |


**Maximum 2 retries** to avoid compounding latency on slow CPU inference (30-110 seconds per call).

### Example Flow

```
User query
   ↓
Retrieval → Context Assembly
   ↓
PromptLoader.build_prompt(context, query)
   ↓
Ollama.generate() → "Sample answer without citations"
   ↓
Validation: FAIL (no [chunk_id])
   ↓
Retry 1: Ollama.generate() → "Valid answer [chunk_001]."
   ↓
Validation: PASS → Return structured response
```

---

## Source Files


| File                                   | Purpose                                                         |
| -------------------------------------- | --------------------------------------------------------------- |
| `configs/prompts/v1.yaml`              | Prompt YAML file                                                |
| `src/paperlens/api/prompt_loader.py`   | `PromptLoader` and `PromptTemplate` classes                     |
| `src/paperlens/api/rag.py`             | `LLMOutput` validation, `_generate_with_fallback()` retry logic |
| `src/paperlens/settings.py`            | `prompts_dir`, `prompt_version` configuration                   |
| `src/paperlens/api/schemas.py`         | `QueryResponse`, `Citation` Pydantic models                     |
| `tests/test_api/test_prompt_loader.py` | Prompt loader tests                                             |
| `tests/test_api/test_rag.py`           | Refusal and retry path tests                                    |


---

## Troubleshooting


| Problem                                           | Cause                                                                        | Fix                                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `FileNotFoundError: Prompt file not found`        | YAML file missing at expected path                                           | Check `PROMPTS_DIR` and `PROMPT_VERSION` settings                 |
| `ValueError: Prompt YAML missing required fields` | YAML lacks `version`, `system_prompt`, `user_template`, or `refusal_message` | Add missing fields to YAML                                        |
| Response lacks citations                          | Prompt not followed, or model not prompted correctly                         | Ensure `{context}` and `[chunk_id]` instructions in system_prompt |
| Refusal returns before retrieval                  | Prompt too aggressive                                                        | Adjust `refusal_message` requirements or system_prompt wording    |
| Tests fail with `Template` error                  | `{context}` or `{query}` placeholder missing                                 | Ensure user_template contains both placeholders                   |
| `configs/prompts/.gitkeep` exists                 | Old placeholder not removed                                                  | Delete `.gitkeep`, ensure `v1.yaml` is tracked                    |


---

*Document created: 03-08-2026*
