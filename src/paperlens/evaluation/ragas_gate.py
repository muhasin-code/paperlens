"""Network-free validation and aggregation for Ragas evaluation results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

_MISSING = object()


def normalize_score(value: object) -> float | None:
    """Return a finite score in [0, 1], or None when the value is invalid."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None

    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None

    return score


def _validate_expected_rows(expected_rows: int) -> None:
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise TypeError("expected_rows must be an integer")
    if expected_rows < 0:
        raise ValueError("expected_rows must be non-negative")


def _row_id(row: object, index: int) -> str:
    if isinstance(row, Mapping):
        value = row.get("id")
        if isinstance(value, str) and value.strip():
            return value
    return f"<row-{index + 1}>"


def summarize_metric(
    rows: Sequence[Mapping[str, object]],
    metric_name: str,
    expected_rows: int,
) -> dict[str, object]:
    """Summarize valid scores while preserving invalid-score coverage.

    The reported mean is explicitly over valid rows only. A gate must require
    complete coverage before using that mean to pass.
    """
    _validate_expected_rows(expected_rows)

    row_count = len(rows)
    id_counts: dict[str, int] = {}
    for _index, row in enumerate(rows):
        if isinstance(row, Mapping):
            value = row.get("id")
            if isinstance(value, str) and value.strip():
                id_counts[value] = id_counts.get(value, 0) + 1

    valid_scores: list[float] = []
    invalid_ids: list[str] = []
    invalid_reasons: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        row_id = _row_id(row, index)
        reasons: list[str] = []

        if index >= expected_rows:
            reasons.append("unexpected_row")

        if not isinstance(row, Mapping):
            reasons.append("invalid_row")
            score_value = _MISSING
        else:
            if row_id.startswith("<row-"):
                reasons.append("missing_id")

            if id_counts.get(row_id, 0) > 1:
                reasons.append("duplicate_id")

            n_contexts = row.get("n_contexts")
            if isinstance(n_contexts, bool) or not isinstance(n_contexts, int) or n_contexts <= 0:
                reasons.append("empty_or_invalid_contexts")

            refused = row.get("refused", _MISSING)
            if refused is True:
                reasons.append("refused_answer")
            elif refused is not _MISSING and not isinstance(refused, bool):
                reasons.append("invalid_refusal_flag")

            score_value = row.get(metric_name, _MISSING)

        score = normalize_score(score_value)
        if score is None:
            reasons.append("missing_or_invalid_score")

        if reasons:
            invalid_ids.append(row_id)
            invalid_reasons.append({"id": row_id, "reason": ",".join(reasons)})
        else:
            valid_scores.append(score)

    for index in range(row_count, expected_rows):
        missing_id = f"<missing-row-{index + 1}>"
        invalid_ids.append(missing_id)
        invalid_reasons.append({"id": missing_id, "reason": "missing_row"})

    n_valid = len(valid_scores)
    n_invalid = max(expected_rows, row_count) - n_valid
    coverage_ratio = n_valid / expected_rows if expected_rows else 0.0

    return {
        "metric": metric_name,
        "mean": sum(valid_scores) / n_valid if n_valid else None,
        "mean_scope": "valid_scores_only",
        "mean_denominator": n_valid,
        "n_expected": expected_rows,
        "n_rows": row_count,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "coverage_ratio": coverage_ratio,
        "row_count_matches_expected": row_count == expected_rows,
        "invalid_ids": invalid_ids,
        "invalid_reasons": invalid_reasons,
    }


def evaluate_faithfulness_gate(
    result: Mapping[str, object],
    threshold: float,
    expected_rows: int,
) -> dict[str, object]:
    """Evaluate the faithfulness threshold and require complete valid coverage."""
    _validate_expected_rows(expected_rows)

    normalized_threshold = normalize_score(threshold)
    if normalized_threshold is None:
        raise ValueError("threshold must be a finite number in [0, 1]")

    n_expected = result.get("n_expected")
    n_rows = result.get("n_rows")
    n_valid = result.get("n_valid")
    n_invalid = result.get("n_invalid")
    mean = normalize_score(result.get("mean"))

    if expected_rows == 0:
        reason = "empty_dataset"
    elif n_expected != expected_rows:
        reason = "expected_row_count_mismatch"
    elif n_rows != expected_rows or result.get("row_count_matches_expected") is not True:
        reason = "row_count_mismatch"
    elif n_valid != expected_rows or n_invalid != 0:
        reason = "insufficient_valid_score_coverage"
    elif mean is None:
        reason = "missing_or_invalid_mean"
    elif mean < normalized_threshold:
        reason = "below_threshold"
    else:
        reason = "passed"

    return {
        "gate_passed": reason == "passed",
        "status": "passed" if reason == "passed" else "failed",
        "reason": reason,
        "threshold": normalized_threshold,
        "n_expected": expected_rows,
        "n_rows": n_rows,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "mean": mean,
        "invalid_ids": result.get("invalid_ids", []),
    }


def evaluation_failure_result(exc: Exception) -> dict[str, object]:
    """Describe an infrastructure/evaluation failure without exposing its text."""
    return {
        "gate_passed": False,
        "status": "infrastructure_error",
        "exit_code": 2,
        "error_type": type(exc).__name__,
    }
