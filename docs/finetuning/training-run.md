# QLoRA Fine-Tuning Run Documentation

Training executed on Kaggle Notebook with T4 GPU (16 GB VRAM).

## Hyperparameter Table

| Parameter                    | Value                              |
|------------------------------|-------------------------------------|
| Base model                   | Qwen/Qwen2.5-3B-Instruct           |
| LoRA rank (r)                | 16                                  |
| LoRA alpha                   | 32                                  |
| LoRA dropout                 | 0.05                                |
| Target modules               | all-linear                          |
| Bias                         | none                                |
| Quantization                 | 4-bit NF4 (bitsandbytes)            |
| Compute / torch dtype        | float16                             |
| Per device train batch size  | 2                                   |
| Gradient accumulation steps  | 8                                   |
| Effective batch size         | 16                                  |
| Learning rate                | 2e-4                                |
| LR scheduler                 | cosine                              |
| Warmup ratio                 | 0.03                                |
| Num train epochs             | 3                                   |
| Max sequence length          | 1024                                |
| Optimizer                    | paged_adamw_32bit                   |
| fp16                         | True                                |
| Loss type                    | nll                                 |
| Logging steps                | 20                                  |
| Save strategy                | epoch                               |
| Load best model at end       | True                                |
| Metric for best model        | eval_loss                           |

## Training Duration

- **Wall clock time on Kaggle T4**: 168.9 minutes (~2.82 hours / 2h 49min)
- **Total training steps**: 435

## Final Training and Validation Loss

| Epoch | Training Loss | Validation Loss |
|-------|---------------|------------------|
| 1     | 1.3305        | 1.3438           |
| 2     | 1.2707        | 1.3225           |
| 3     | 1.1935        | 1.3249           |

- **Final overall train loss** (mean across full run, as returned by `trainer.train()`): 1.3395
- **Best checkpoint**: epoch 2 (`checkpoint-290`), selected automatically via `load_best_model_at_end=True` / `metric_for_best_model="eval_loss"` — validation loss rose slightly at epoch 3 (1.3225 → 1.3249) while training loss kept falling, a mild overfitting signal, so the epoch-2 weights are what was actually saved and pushed, not epoch 3's.

## Loss Curve

```
Epoch 1: training_loss_epoch_1 = 1.3305
Epoch 2: training_loss_epoch_2 = 1.2707
Epoch 3: training_loss_epoch_3 = 1.1935
```

## GPU Memory Usage

- **Quantized base model (4-bit NF4)**: ~4.5 GB (planned estimate)
- **LoRA adapter parameters**: ~0.3 GB (planned estimate)
- **Optimizer states (paged_adamw)**: ~1.5 GB (planned estimate)
- **Activations at batch size 2, seq len 1024**: ~4 GB (planned estimate; note actual per-device batch size was 2, not the originally planned 4 — see Deviations)
- **Total peak (planned estimate)**: ~10-11 GB (fits in T4 16 GB with ~5 GB headroom)
- **Actual peak from nvidia-smi**: TBD — not captured during this run. If you have it from Kaggle's session GPU-usage graph, share it and I'll fill this in.

## Deviations from Planned Hyperparameters

- **Per-device train batch size / gradient accumulation**: planned as 4 / 4; actual run used **2 / 8**. Effective batch size is unchanged (16), but the smaller per-step batch trades more accumulation steps for lower per-step activation memory — a safer margin on the T4's 16 GB.
- **Logging steps**: planned as 10; actual run used **20**.
- **Compute/torch dtype**: not specified in the original plan, but this required an explicit decision during the run. The notebook initially used `torch.bfloat16` for both `bnb_4bit_compute_dtype` and `torch_dtype`, which is incompatible with the T4 (Turing architecture, compute capability 7.5, no native bf16 tensor-core support). Switched to **`torch.float16`** throughout to match `fp16=True` mixed-precision training.
- **`loss_type="nll"` added explicitly** (not in the original hyperparameter set). TRL's newer default `chunked_nll` loss path crashed at trainer construction (`AttributeError` while patching the model's `lm_head` forward) specifically in combination with `device_map="auto"`. Setting `loss_type="nll"` restores the standard, non-chunked cross-entropy path.
- **LoRA adapter weight dtype fix**: despite the base model loading correctly in `float16`, all 504 trainable LoRA adapter parameters (`lora_A`/`lora_B` across all 36 layers × 7 target modules) initialized as `bfloat16`, which crashed `GradScaler` (`fp16=True` requires float16 gradients). Fixed by explicitly casting all trainable parameters to `float32` immediately before `trainer.train()` in the same cell, so no intervening step could silently revert it.

## HuggingFace Hub Model ID

- **Repository**: `muhasin-code/paperlens-qwen2.5-3b-extraction` *(corrected — originally written as `muhasin/paperlens-qwen2.5-3b-extraction`; actual namespace is `muhasin-code`, matching the authenticated HF account)*
- **Adapter files**: `adapter_config.json`, `adapter_model.safetensors` *(corrected — PEFT's current default save format is `.safetensors`, not `.bin`)*
- **Repo visibility**: private
- **Upload size**: ~120 MB (per-epoch `checkpoint-*/` directories deliberately excluded from the push to stay within adapter-only footprint)

## Dataset Information

- **Training samples**: 2,309
- **Validation samples**: 256
- **Source (repo)**: `data/extraction_dataset/train.jsonl` and `val.jsonl`
- **Source (as loaded on Kaggle)**: `/kaggle/input/datasets/muhammedmuhasink/paperlens-extraction-dataset/{train,val}.jsonl` (Kaggle Dataset mirror of the repo data)

## Smoke Test Results (post-push, loaded fresh from Hub)

Ran against 3 held-out validation examples, loading the pushed adapter fresh from the Hub (not the in-memory training session) to verify the shipped artifact works standalone:

- **Valid JSON output**: 3/3
- **Schema correctness**: all 3 outputs contained the expected 5 fields (`research_question`, `method`, `datasets`, `metrics`, `key_finding`)
- **Exact field-value match vs. ground truth**: `method` 3/3, `datasets` 2/3, `research_question`/`metrics`/`key_finding` 0/3
  - Note: this is a strict string-equality check, not a semantic-correctness eval — 0/3 exact matches on free-text fields like `research_question` and `key_finding` may reflect valid paraphrasing rather than genuine errors. Not yet diagnosed with a side-by-side predicted-vs-ground-truth comparison; treat this as a structural smoke test pass, not a quality evaluation.

---

*Document created after Kaggle training completion.*
