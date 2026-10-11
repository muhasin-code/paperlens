#!/usr/bin/env python3
"""Evaluate fine-tuned extraction model vs base model.

Usage:
    python scripts/evaluate_extraction.py [--limit N] [--output PATH]
    make evaluate-extraction

This script:
- Loads base model (Qwen/Qwen2.5-3B-Instruct) and fine-tuned adapter from HF Hub
- Loads test set from data/extraction_dataset/val.jsonl
- Runs inference on both models with the same prompts
- Computes JSON validity rate, field-level F1, and exact match accuracy
- Outputs results as JSON and prints formatted summary table
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.extraction.annotator import DEFAULT_SYSTEM_PROMPT
from src.paperlens.settings import get_settings

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None


REQUIRED_FIELDS = ["research_question", "method", "datasets", "metrics", "key_finding"]


def load_val_dataset(path: Path) -> list[dict[str, Any]]:
    """Load validation dataset from JSONL file."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if "messages" in row:
                        examples.append(row)
                except json.JSONDecodeError:
                    continue
    return examples


def extract_assistant_response(text: str) -> str:
    """Extract assistant response from model output."""
    if "assistant" in text:
        parts = text.split("assistant")
        if len(parts) > 1:
            text = parts[-1].strip()
    text = text.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json_match.group(0)
    return text


def parse_output(text: str) -> dict[str, Any] | None:
    """Parse model output as JSON, returning None if invalid."""
    try:
        data = json.loads(text)
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def is_valid_output(data: dict | None) -> bool:
    """Check if output has all 5 required fields."""
    if data is None:
        return False
    return all(field in data for field in REQUIRED_FIELDS)


def compute_field_f1(
    predicted: Any, ground_truth: Any, field_name: str
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for a field.

    For list fields (datasets, metrics): use set overlap.
    For string fields: use token-level comparison.
    """
    if predicted is None and ground_truth is None:
        return 1.0, 1.0, 1.0
    if predicted is None or ground_truth is None:
        return 0.0, 0.0, 0.0

    if field_name in ("datasets", "metrics"):
        pred_set = set(predicted) if isinstance(predicted, list) else {predicted}
        truth_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}

        if not pred_set and not truth_set:
            return 1.0, 1.0, 1.0

        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    else:
        pred_str = str(predicted).strip().lower()
        truth_str = str(ground_truth).strip().lower()

        if pred_str == truth_str:
            return 1.0, 1.0, 1.0

        pred_tokens = set(pred_str.split())
        truth_tokens = set(truth_str.split())

        if not pred_tokens and not truth_tokens:
            return 1.0, 1.0, 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0, 0.0, 0.0

        tp = len(pred_tokens & truth_tokens)
        fp = len(pred_tokens - truth_tokens)
        fn = len(truth_tokens - pred_tokens)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1


def compute_exact_match(predicted: Any, ground_truth: Any) -> bool:
    """Compute exact match (normalized)."""
    if predicted is None and ground_truth is None:
        return True
    if predicted is None or ground_truth is None:
        return False

    if isinstance(predicted, list) and isinstance(ground_truth, list):
        pred_norm = sorted([str(p).strip().lower() for p in predicted])
        truth_norm = sorted([str(t).strip().lower() for t in ground_truth])
        return pred_norm == truth_norm
    else:
        pred_norm = str(predicted).strip().lower()
        truth_norm = str(ground_truth).strip().lower()
        return pred_norm == truth_norm


def evaluate_model(model, tokenizer, examples: list[dict], system_prompt: str) -> dict[str, Any]:
    """Evaluate a model on the extraction task."""
    total = len(examples)
    valid_json_count = 0
    field_stats = {field: {"tp": 0, "fp": 0, "fn": 0} for field in REQUIRED_FIELDS}
    exact_match_method = 0
    exact_match_key_finding = 0
    examples_details = []

    for i, example in enumerate(examples):
        messages = example.get("messages", [])
        if not messages or len(messages) < 2:
            continue

        system = messages[0].get("content", system_prompt)
        user_content = messages[1].get("content", "")

        prompt = f"{system}\n\n{user_content}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_text = extract_assistant_response(response)
        predicted = parse_output(assistant_text)

        is_valid = is_valid_output(predicted)
        if is_valid:
            valid_json_count += 1

        for field in REQUIRED_FIELDS:
            pred_val = predicted.get(field) if predicted else None
            truth_val = example.get("metadata", {}).get(field)
            if truth_val is None:
                truth_val = None

            p, r, f1 = compute_field_f1(pred_val, truth_val, field)
            field_stats[field]["tp"] += p
            field_stats[field]["fp"] += 1 - p
            field_stats[field]["fn"] += 1 - r

        if is_valid and predicted:
            if compute_exact_match(
                predicted.get("method"), example.get("metadata", {}).get("method")
            ):
                exact_match_method += 1
            if compute_exact_match(
                predicted.get("key_finding"), example.get("metadata", {}).get("key_finding")
            ):
                exact_match_key_finding += 1

        examples_details.append(
            {
                "index": i,
                "valid_json": is_valid,
                "predicted": predicted,
                "ground_truth": {f: example.get("metadata", {}).get(f) for f in REQUIRED_FIELDS},
            }
        )

    field_f1 = {}
    for field in REQUIRED_FIELDS:
        stats = field_stats[field]
        precision = (
            stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        )
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        field_f1[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total_examples": total,
        "json_valid_rate": valid_json_count / total if total > 0 else 0,
        "field_f1": field_f1,
        "exact_match_method": exact_match_method / total if total > 0 else 0,
        "exact_match_key_finding": exact_match_key_finding / total if total > 0 else 0,
        "examples": examples_details,
    }


async def run_evaluation_async(
    base_model_id: str,
    adapter_repo: str,
    examples: list[dict],
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation using HuggingFace Inference API for slower inference."""
    if torch is not None and torch.cuda.is_available():
        return await run_evaluation_local(base_model_id, adapter_repo, examples, limit)
    else:
        return await run_evaluation_hf_api(base_model_id, adapter_repo, examples, limit)


async def run_evaluation_local(
    base_model_id: str,
    adapter_repo: str,
    examples: list[dict],
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation using local model with transformers + PEFT."""
    print("Loading base model locally...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()

    if limit:
        examples = examples[:limit]

    system_prompt = DEFAULT_SYSTEM_PROMPT

    results = evaluate_model(model, tokenizer, examples, system_prompt)

    print(f"Base model + adapter local evaluation complete: {results['total_examples']} examples")
    return results


async def run_evaluation_hf_api(
    base_model_id: str,
    adapter_repo: str,
    examples: list[dict],
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation using HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    if limit:
        examples = examples[:limit]

    print("Using HuggingFace Inference API for evaluation...")
    client = InferenceClient()

    base_results = []
    for i, example in enumerate(examples):
        messages = example.get("messages", [])
        if not messages or len(messages) < 2:
            continue
        prompt = f"{messages[0].get('content', '')}\n\n{messages[1].get('content', '')}"
        try:
            response = client.text_generation(prompt, max_new_tokens=256, temperature=0.1)
            base_results.append(
                {
                    "index": i,
                    "response": response,
                    "example": example,
                }
            )
        except Exception as e:
            print(f"API call failed for example {i}: {e}")
            continue

    adapter_results = []
    for i, example in enumerate(examples):
        messages = example.get("messages", [])
        if not messages or len(messages) < 2:
            continue
        prompt = f"{messages[0].get('content', '')}\n\n{messages[1].get('content', '')}"
        try:
            response = client.text_generation(
                prompt, max_new_tokens=256, temperature=0.1, model=adapter_repo
            )
            adapter_results.append(
                {
                    "index": i,
                    "response": response,
                    "example": example,
                }
            )
        except Exception as e:
            print(f"Adapter API call failed for example {i}: {e}")
            continue

    return {
        "api_base_results": base_results,
        "api_adapter_results": adapter_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 10 examples only")
    args = parser.parse_args()

    settings = get_settings()

    val_path = settings.extraction_val_path
    if not val_path.exists():
        print(f"Error: Validation file not found: {val_path}", file=sys.stderr)
        return 1

    examples = load_val_dataset(val_path)
    if not examples:
        print("Error: No valid examples found in validation set", file=sys.stderr)
        return 1

    if args.dry_run:
        examples = examples[:10]
    if args.limit:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} examples from {val_path}")

    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    adapter_repo = settings.extraction_adapter_repo or settings.hf_hub_model_id

    print(f"\nBase model: {base_model_id}")
    print(f"Adapter: {adapter_repo}")

    results = asyncio.run(run_evaluation_async(base_model_id, adapter_repo, examples))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal examples: {results.get('total_examples', len(examples))}")
    print(f"\nJSON Validity Rate: {results.get('json_valid_rate', 0) * 100:.1f}%")

    print("\nField-Level F1:")
    print("-" * 40)
    for field, stats in results.get("field_f1", {}).items():
        print(
            f"  {field}: P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1']:.3f}"
        )

    print("\nExact Match Accuracy:")
    print("-" * 40)
    print(f"  method: {results.get('exact_match_method', 0) * 100:.1f}%")
    print(f"  key_finding: {results.get('exact_match_key_finding', 0) * 100:.1f}%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
