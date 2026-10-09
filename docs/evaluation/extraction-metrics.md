# Extraction Quality Benchmark

Evaluation of the QLoRA fine-tuned Qwen2.5-3B-Instruct adapter on the PaperLens structured-extraction task.

**Status: not passed.** The adapter produces valid JSON about 90% of the time, but on three of five fields it scores below a trivial "always null" baseline. Part of the cause appears to be the training labels, not only the model (see [Label Audit](#label-audit-training-data)). All numbers below were measured on the 256-example validation set unless stated otherwise.

## Evaluation Setup

- **Base model**: Qwen/Qwen2.5-3B-Instruct
- **Fine-tuned adapter**: muhasin-code/paperlens-qwen2.5-3b-extraction
- **Test set**: `data/extraction_dataset/val.jsonl` (256 examples). Training set: 2,309 examples.
- **Training**: LoRA r=16, alpha=32, dropout 0.05, target_modules=all-linear; 4-bit NF4; 3 epochs; lr 2e-4 (cosine); effective batch size 16; max_length 1024. Best checkpoint was epoch 2 (val loss 1.3225). Per-epoch train/val loss: 1.3305/1.3438, 1.2707/1.3225, 1.1935/1.3249.
- **Loss masking**: the training notebook formats each example into one chat-template text field and sets no completion-only/assistant-only loss option, so the loss appears to cover system prompt, chunk and answer tokens alike.
- **System prompt**: `DEFAULT_SYSTEM_PROMPT` from `src/paperlens/extraction/annotator.py` (carried in each val row).
- **Inference config**: chat-template prompt with generation prompt, greedy decoding (`do_sample=False`, so no temperature), `repetition_penalty=1.0`, `max_new_tokens=256`, batch size 16. Chunk text is capped at 900 tokens (never triggered; longest chunk is 599 tokens).
- **Environment**: Kaggle Tesla T4, 4-bit NF4 with fp16 compute, SDPA attention; transformers 5.0.0, peft 0.19.1, torch 2.10.0+cu128, bitsandbytes 0.50.2.
- **Prompt-format check** (24 examples): chat template gave 96% valid JSON and mean F1 0.493; a raw `system + user` string gave 0% valid JSON. The chat template matches how the adapter was trained.
- **Base model**: **not evaluated**, so there are no base-vs-fine-tuned comparisons in this document. The reference point used instead is an all-null baseline (below).
- **Code**: `evaluate_extraction.py`; raw results in `extraction_eval_results.json`.

### Scoring definitions

- Each field is scored per example, then averaged over all 256 examples (macro average). Mean F1 is the average of per-example F1, so it is not the F1 of mean precision and recall.
- Text fields use token-set F1 (lowercased, whitespace-split). List fields (`datasets`, `metrics`) use set F1 over lowercased items.
- Both-null counts as a perfect match. For list fields, `[]` and `null` are treated as equivalent. If exactly one side is null, the score is 0.
- Output that fails to parse as JSON scores 0 on every field. This is deliberate, but it lowers all averages by up to about 10%.
- Exact match compares normalized (stripped, lowercased) values; list fields compare as sets.

## JSON Validity Rate

Percentage of outputs that parse as valid JSON containing all 5 required keys.

| Model | JSON Validity Rate |
|-------|-------------------|
| Base (Qwen2.5-3B-Instruct) | Not evaluated |
| Fine-tuned (LoRA adapter) | **89.8%** (230/256) |

An approximate 95% interval (normal approximation) is about 86% to 94%. A 50-example dry run on the first 50 rows showed 94.0%, which was optimistic.

## Field-Level F1 Scores

Fine-tuned adapter on all 256 validation examples, next to an all-null baseline (predict `null` for every field) scored the same way.

| Field | Precision | Recall | F1 | All-null baseline F1 |
|-------|-----------|--------|----|----------------------|
| research_question | 0.235 | 0.140 | 0.168 | 0.000 |
| method | 0.735 | 0.753 | 0.743 | 0.797 |
| datasets | 0.652 | 0.652 | 0.652 | 0.711 |
| metrics | 0.732 | 0.676 | 0.676 | 0.020 |
| key_finding | 0.031 | 0.027 | 0.024 | 0.543 |

The adapter beats the baseline on `research_question` and `metrics`, and is below it on `method`, `datasets` and `key_finding`. Because invalid outputs score zero, part of the gap on `method` and `datasets` comes from the 26 invalid outputs. The baseline has no failures. I have not computed the baseline on the 230 parsed examples only.

### Null behavior

| Field | Truth null (parsed examples) | Model predicts null |
|-------|------------------------------|---------------------|
| research_question | 0% | 0% |
| method | 81% | 82% |
| datasets | 72% | 100% |
| metrics | 2% | 0% |
| key_finding | 53% | 0% |

The model never predicts a `key_finding` null, although about half the references are null, and it predicts no `datasets` for every parsed output, although about 28% of references have one.

## Exact Match Accuracy

Percentage of outputs where the predicted value equals the ground truth after normalization. Both-null counts as a match, so `method` and `datasets` are inflated by the many null references.

| Field | Fine-tuned |
|-------|------------|
| research_question | 0.0% |
| method | 67.6% |
| datasets | 65.2% |
| metrics | 16.0% |
| key_finding | 0.0% |

## Examples (Fine-tuned Output vs. Ground Truth)

These are the first validation examples from the 50-example dry run; no base-model output exists. Input chunks are omitted.

### Example 1 (val row 0)

**Fine-tuned output**:
```json
{
  "research_question": "What is the baseline?",
  "method": null,
  "datasets": null,
  "metrics": ["AR"],
  "key_finding": "The baseline is a policy that randomly selects an action."
}
```

**Ground truth**:
```json
{
  "research_question": "How is data for RTFM collected?",
  "method": null,
  "datasets": null,
  "metrics": ["loss"],
  "key_finding": "Proposed model achive 66+-22 win rate, baseline CNN 13+-1  and baseline FiLM 32+-3 ."
}
```

### Examples 2 to 4 (val rows 1, 2, 4; selected fields)

| Row | Field | Fine-tuned output | Ground truth |
|-----|-------|-------------------|--------------|
| 1 | research_question | What is the baseline model? | Which baselines did they compare? |
| 1 | key_finding | Answer with content missing: (Table 1) The baseline model is a deep sequence-to-sequence encoder/decoder model with attention. | one |
| 2 | research_question | What is the baseline? | Do they normalize the calculated intermediate output hypotheses to compensate for the incompleteness? |
| 2 | key_finding | Answer with content missing: (Table 1) The proposed model achieves 1.0% relative improvement over the best baseline on Librispeech and 0.5% relative improvement on the video dataset. | None |
| 4 | research_question | What is the baseline? | In which setting they achieve the state of the art? |
| 4 | key_finding | Answer with content missing: (Table 1 ) Accuracy of the model with basic question (s1, s2, s3) on the test set. | in open-ended task esp. for counting-type questions |

## Failure Mode Analysis

Observed failure modes of the fine-tuned adapter. There is no base-model comparison because the base model was not evaluated.

| Failure mode | Evidence | Scale |
|--------------|----------|-------|
| Always predicts no `datasets` | Null in 100% of parsed outputs vs 72% of references | All parsed outputs |
| Never predicts null `key_finding` | 0% null vs 53% of references | All parsed outputs |
| QA-style filler in `key_finding` | Starts with "Answer with content missing" | 85 of 230 parsed outputs (37%); about 1% of val and 2% of train labels |
| Generic `research_question` | "What is the baseline?" was the most common output | 20 of 47 parsed outputs in the first-50 subset |
| Run-on output until the token cap | JSON starts correctly, then copies passage text inside `method` or `key_finding` until 256 tokens | 26 of 256 invalid (10.2%); I inspected only 3 of them (rows 3, 32, 48 of the dry run) |

## Label Audit (Training Data)

Checks on all 2,309 training rows, run to understand the results above.

| Check | Result |
|-------|--------|
| Null rate, train vs val | method 77% vs 80%; datasets 78% vs 71%; metrics 2% vs 2%; key_finding 51% vs 54%; research_question 0% vs 0% |
| `research_question` | 100% end with "?"; most common: "do they report results only on english data?" (38x), "which dataset do they use?" (16x) |
| `method` | 525 non-null (23%); 100% of them start with text copied from the chunk |
| `key_finding` | 1,124 non-null (49%); 2% start with copied chunk text |
| `metrics` | 35 distinct strings; AR appears in 2,200 rows (95%), AP in 1,657 (72%), F1 in 1,056; only 17% of metric labels (1,374 of 8,115) appear in their chunk |
| Chunk format | 58% of chunks look like stringified Python lists |

What this suggests (interpretation, not confirmed):

- The `research_question` and `key_finding` labels resemble questions and answers from a QA dataset over NLP papers, and most questions cannot be inferred from the chunk.
- The `metrics` F1 of 0.676 may reflect the model learning which metric names are common (a constant-prior predictor has not been tested).
- Which script produced `train.jsonl` has not been confirmed. The sample rows do not look like output of the LLM annotator in `annotator.py`.

## Latency

| Inference method | Model | Time per example | Notes |
|------------------|-------|------------------|-------|
| Kaggle T4, batched (16), 4-bit | Fine-tuned | 2.21 s | Full 256-example run took 565 s, `max_new_tokens=256` |
| HF Inference API | Base / Fine-tuned | Not measured | |
| Local CPU | Base / Fine-tuned | Not measured | |

## Statistical Significance

- **Sample size**: 256 validation examples.
- **JSON validity**: approximate 95% interval of 86% to 94% (normal approximation).
- **Not computed**: bootstrap intervals for field scores and a paired test against the base model (no base results exist).
- Treat the results as directional.

## Conclusion

The adapter reliably produces JSON in the expected schema (89.8% valid), but it does not yet extract useful structured information. It is below an all-null baseline on `method`, `datasets` and `key_finding`, and its advantage on `research_question` is small (F1 0.168). It beats the baseline on `metrics`, but that may be label-prior matching.

Likely contributors are label provenance (see the audit) and a training loss that appears to include prompt tokens. Neither has been confirmed or ruled out experimentally.

### Suggested next steps

1. Confirm how `train.jsonl` was built and relabel with an LLM that follows the system prompt if the labels are QA-derived.
2. Retrain with loss on the answer tokens only.
3. Add a constant-prior baseline for `metrics` and `datasets`, and evaluate the base model with the same script.
4. Re-run this benchmark and add bootstrap intervals.

---

*Benchmark generated: 2026-10-09*
