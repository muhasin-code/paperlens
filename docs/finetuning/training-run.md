# QLoRA Fine-Tuning Run Documentation

Training executed on Kaggle Notebook with T4 GPU (16 GB VRAM).

## Hyperparameter Table

| Parameter                    | Value                    |
|------------------------------|--------------------------|
| Base model                   | Qwen/Qwen2.5-3B-Instruct |
| LoRA rank (r)                | 16                       |
| LoRA alpha                   | 32                       |
| LoRA dropout                 | 0.05                     |
| Target modules               | all-linear               |
| Bias                         | none                     |
| Quantization                 | 4-bit NF4 (bitsandbytes) |
| Per device train batch size  | 4                        |
| Gradient accumulation steps  | 4                        |
| Effective batch size         | 16                       |
| Learning rate                | 2e-4                     |
| LR scheduler                 | cosine                   |
| Warmup ratio                 | 0.03                     |
| Num train epochs             | 3                        |
| Max sequence length          | 1024                     |
| Optimizer                    | paged_adamw_32bit        |
| fp16                         | True                     |
| Logging steps                | 10                       |
| Save strategy                | epoch                    |
| Load best model at end       | True                     |
| Metric for best model        | eval_loss                |

## Training Duration

- **Wall clock time on Kaggle T4**: TBD hours (to be filled after training)

## Final Training and Validation Loss

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | TBD           | TBD             |
| 2     | TBD           | TBD             |
| 3     | TBD           | TBD             |

## Loss Curve

```

Epoch 1: training_loss_epoch_1 = TBD
Epoch 2: training_loss_epoch_2 = TBD
Epoch 3: training_loss_epoch_3 = TBD

```

## GPU Memory Usage

- **Quantized base model (4-bit NF4)**: ~4.5 GB
- **LoRA adapter parameters**: ~0.3 GB
- **Optimizer states (paged_adamw)**: ~1.5 GB
- **Activations at batch size 4, seq len 1024**: ~4 GB
- **Total peak**: ~10-11 GB (fits in T4 16 GB with ~5 GB headroom)
- **Actual peak from nvidia-smi**: TBD GB

## Deviations from Planned Hyperparameters

TBD - Document any changes made during training.

## HuggingFace Hub Model ID

- **Repository**: `muhasin/paperlens-qwen2.5-3b-extraction`
- **Adapter files**: `adapter_model.bin`, `adapter_config.json`

## Dataset Information

- **Training samples**: TBD
- **Validation samples**: TBD
- **Source**: `data/extraction_dataset/train.jsonl` and `val.jsonl`

---

*Document created after Kaggle training completion.*
