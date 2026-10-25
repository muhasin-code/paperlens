"""Gold-dataset load and schema checks. No Chroma, no models."""

from __future__ import annotations

import json
from pathlib import Path

from src.paperlens.evaluation.metrics import chunk_to_arxiv_id

DATASET_VERSION = "4.3.0"
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
CATEGORIES = frozenset({"concept", "comparison", "architecture", "training", "generation"})


def load_gold_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_retrieval_row(row: dict) -> bool:
    """Missing eval_sets stays in the retrieval benchmark (legacy 26-row file)."""
    sets = row.get("eval_sets")
    if sets is None:
        return True
    return "retrieval" in sets


def is_generation_row(row: dict) -> bool:
    sets = row.get("eval_sets")
    if not sets:
        return False
    return "generation" in sets


def with_expanded_labels(
    entry: dict,
    chunk_ids: list[str],
    arxiv_ids: list[str],
) -> dict:
    """Copy entry and replace label lists. Drops the unused expected_answer_summary key."""
    expanded = dict(entry)
    expanded.pop("expected_answer_summary", None)
    expanded["relevant_chunk_ids"] = list(chunk_ids)
    expanded["relevant_arxiv_ids"] = list(arxiv_ids)
    return expanded


def _chunk_errors(ids: list, *, field: str, index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(ids, list) or not ids:
        return [f"row {index}: {field} must be a non-empty list"]
    for chunk_id in ids:
        if not isinstance(chunk_id, str):
            errors.append(f"row {index}: {field} contains a non-string")
            continue
        try:
            chunk_to_arxiv_id(chunk_id)
        except ValueError:
            errors.append(f"row {index}: {field} is not a PaperLens chunk id: {chunk_id}")
    return errors


def validate_row(row: dict, *, index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get("query"), str) or not row["query"].strip():
        errors.append(f"row {index}: query is empty")
    if row.get("dataset_version") != DATASET_VERSION:
        errors.append(f"row {index}: dataset_version must be {DATASET_VERSION}")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        errors.append(f"row {index}: id is empty")

    errors.extend(
        _chunk_errors(row.get("relevant_chunk_ids"), field="relevant_chunk_ids", index=index)
    )
    arxiv_ids = row.get("relevant_arxiv_ids")
    if not isinstance(arxiv_ids, list) or not arxiv_ids:
        errors.append(f"row {index}: relevant_arxiv_ids must be a non-empty list")

    sets = row.get("eval_sets")
    if not isinstance(sets, list) or not sets:
        errors.append(f"row {index}: eval_sets must be a non-empty list")
    elif any(name not in {"retrieval", "generation"} for name in sets):
        errors.append(f"row {index}: eval_sets has an unknown name {sets}")

    if is_generation_row(row):
        if not isinstance(row.get("ground_truth"), str) or not row["ground_truth"].strip():
            errors.append(f"row {index}: ground_truth is empty")
        errors.extend(
            _chunk_errors(row.get("source_chunk_ids"), field="source_chunk_ids", index=index)
        )
        relevant = set(row.get("relevant_chunk_ids") or [])
        for chunk_id in row.get("source_chunk_ids") or []:
            if isinstance(chunk_id, str) and chunk_id not in relevant:
                errors.append(
                    f"row {index}: source_chunk_id {chunk_id} is not in relevant_chunk_ids"
                )

    difficulty = row.get("difficulty")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        errors.append(f"row {index}: difficulty {difficulty!r} is not allowed")
    category = row.get("category")
    if category is not None and category not in CATEGORIES:
        errors.append(f"row {index}: category {category!r} is not allowed")
    return errors


def validate_dataset(rows: list[dict], *, min_generation: int | None) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for index, row in enumerate(rows, start=1):
        errors.extend(validate_row(row, index=index))
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                errors.append(f"row {index}: duplicate id {row_id}")
            seen_ids.add(row_id)
        query = row.get("query")
        if isinstance(query, str):
            if query in seen_queries:
                errors.append(f"row {index}: duplicate query")
            seen_queries.add(query)
    if min_generation is not None:
        count = sum(1 for row in rows if is_generation_row(row))
        if count < min_generation:
            errors.append(f"generation rows {count} < {min_generation}")
    return errors
