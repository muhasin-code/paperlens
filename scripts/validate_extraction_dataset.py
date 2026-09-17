#!/usr/bin/env python3
"""Validate the extraction dataset integrity.

Usage:
    python scripts/validate_extraction_dataset.py
    make validate-extraction-dataset

This script checks:
- train.jsonl and val.jsonl exist and are non-empty
- Every row has the required messages structure
- Assistant message is valid JSON with extraction fields
- No chunk_id appears in both train and val splits
- Train/val ratio is within 5% of configured split
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.settings import get_settings


def validate_file(path: Path) -> tuple[bool, int, list[str]]:
    """Validate a single JSONL file. Returns (is_valid, row_count, errors)."""
    errors = []

    if not path.exists():
        return False, 0, [f"File not found: {path}"]

    if path.stat().st_size == 0:
        return False, 0, [f"File is empty: {path}"]

    row_count = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
                continue

            if "messages" not in row:
                errors.append(f"Line {i}: Missing 'messages' key")
                continue

            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) != 3:
                errors.append(f"Line {i}: 'messages' must have exactly 3 elements")
                continue

            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"Line {i}: Invalid message roles: {roles}")
                continue

            assistant_msg = messages[2].get("content", "")
            try:
                output = json.loads(assistant_msg)
                required_fields = [
                    "research_question",
                    "method",
                    "datasets",
                    "metrics",
                    "key_finding",
                ]
                missing = [f for f in required_fields if f not in output]
                if missing:
                    errors.append(f"Line {i}: Missing output fields: {missing}")
            except json.JSONDecodeError:
                errors.append(f"Line {i}: Assistant message is not valid JSON")

            row_count += 1

    return len(errors) == 0, row_count, errors


def validate_split(
    train_chunk_ids: set[str], val_chunk_ids: set[str], target_ratio: float, tolerance: float = 0.05
) -> tuple[bool, str]:
    """Validate train/val split ratio and check for overlap."""
    if train_chunk_ids & val_chunk_ids:
        overlap = train_chunk_ids & val_chunk_ids
        return False, f"Overlap in chunk_ids: {len(overlap)} chunks in both splits"

    total = len(train_chunk_ids) + len(val_chunk_ids)
    if total == 0:
        return False, "No examples in dataset"

    actual_val_ratio = len(val_chunk_ids) / total
    diff = abs(actual_val_ratio - target_ratio)

    if diff > tolerance:
        return (
            False,
            f"Val ratio {actual_val_ratio:.3f} differs from target {target_ratio:.3f} by {diff:.3f}",
        )

    return (
        True,
        f"Train/val split valid: {len(train_chunk_ids)} train, {len(val_chunk_ids)} val ({actual_val_ratio:.1%} val)",
    )


def collect_chunk_ids(path: Path) -> set[str]:
    """Extract all chunk_ids from a JSONL file."""
    chunk_ids = set()
    if not path.exists():
        return chunk_ids

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "metadata" in row and "chunk_id" in row["metadata"]:
                    chunk_ids.add(row["metadata"]["chunk_id"])
            except json.JSONDecodeError:
                pass

    return chunk_ids


def main() -> int:
    settings = get_settings()

    print("Validating extraction dataset...")
    print(f"Train path: {settings.extraction_train_path}")
    print(f"Val path: {settings.extraction_val_path}")
    print(f"Target val split: {settings.extraction_val_split}")

    all_passed = True

    print("\n=== Validating train.jsonl ===")
    train_valid, train_count, train_errors = validate_file(settings.extraction_train_path)
    print(f"  Valid: {train_valid}")
    print(f"  Rows: {train_count}")
    if train_errors:
        print(f"  Errors: {len(train_errors)}")
        for err in train_errors[:5]:
            print(f"    - {err}")
        if len(train_errors) > 5:
            print(f"    ... and {len(train_errors) - 5} more")
        all_passed = False

    print("\n=== Validating val.jsonl ===")
    val_valid, val_count, val_errors = validate_file(settings.extraction_val_path)
    print(f"  Valid: {val_valid}")
    print(f"  Rows: {val_count}")
    if val_errors:
        print(f"  Errors: {len(val_errors)}")
        for err in val_errors[:5]:
            print(f"    - {err}")
        if len(val_errors) > 5:
            print(f"    ... and {len(val_errors) - 5} more")
        all_passed = False

    print("\n=== Validating train/val split ===")
    train_chunk_ids = collect_chunk_ids(settings.extraction_train_path)
    val_chunk_ids = collect_chunk_ids(settings.extraction_val_path)

    split_valid, split_msg = validate_split(
        train_chunk_ids, val_chunk_ids, settings.extraction_val_split
    )
    print(f"  Valid: {split_valid}")
    print(f"  {split_msg}")
    if not split_valid:
        all_passed = False

    print("\n=== Summary ===")
    print(f"  train.jsonl: {'PASS' if train_valid else 'FAIL'} ({train_count} rows)")
    print(f"  val.jsonl: {'PASS' if val_valid else 'FAIL'} ({val_count} rows)")
    print(f"  Split: {'PASS' if split_valid else 'FAIL'}")
    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
