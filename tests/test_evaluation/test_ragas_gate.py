import math

import pytest

from src.paperlens.evaluation.ragas_gate import (
    evaluate_faithfulness_gate,
    evaluation_failure_result,
    normalize_score,
    summarize_metric,
)


def _rows(scores: list[object]) -> list[dict[str, object]]:
    return [
        {
            "id": f"qa_{index + 1:03}",
            "n_contexts": 2,
            "refused": False,
            "faithfulness": score,
        }
        for index, score in enumerate(scores)
    ]


def _gate(scores: list[object]) -> dict[str, object]:
    summary = summarize_metric(_rows(scores), "faithfulness", len(scores))
    return evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=len(scores))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (0.75, 0.75),
        (1, 1.0),
        (None, None),
        ("NaN", None),
        (True, None),
        (-0.01, None),
        (1.01, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
    ],
)
def test_normalize_score(value, expected):
    assert normalize_score(value) == expected


def test_below_threshold_fails():
    result = _gate([0.74, 0.74])
    assert result["gate_passed"] is False
    assert result["reason"] == "below_threshold"


def test_exactly_at_threshold_passes_with_full_coverage():
    result = _gate([0.75, 0.75])
    assert result["gate_passed"] is True
    assert result["mean"] == 0.75


def test_above_threshold_passes():
    result = _gate([0.8, 0.9])
    assert result["gate_passed"] is True


@pytest.mark.parametrize(
    "invalid_score",
    [
        pytest.param(None, id="null"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param("NaN", id="nan-string"),
        pytest.param(True, id="boolean"),
        pytest.param(1.1, id="above-range"),
    ],
)
def test_invalid_score_fails_coverage(invalid_score):
    summary = summarize_metric(
        _rows([0.9, invalid_score]),
        "faithfulness",
        expected_rows=2,
    )
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=2)

    assert result["gate_passed"] is False
    assert result["reason"] == "insufficient_valid_score_coverage"
    assert summary["n_valid"] == 1
    assert summary["n_invalid"] == 1


def test_missing_score_fails_coverage():
    rows = _rows([0.9, 0.9])
    del rows[1]["faithfulness"]

    summary = summarize_metric(rows, "faithfulness", expected_rows=2)
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=2)

    assert result["gate_passed"] is False
    assert summary["n_valid"] == 1
    assert summary["invalid_ids"] == ["qa_002"]


def test_missing_row_fails_coverage():
    summary = summarize_metric(_rows([0.9]), "faithfulness", expected_rows=2)
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=2)

    assert result["gate_passed"] is False
    assert summary["n_valid"] == 1
    assert summary["n_invalid"] == 1
    assert "<missing-row-2>" in summary["invalid_ids"]


def test_empty_dataset_fails():
    summary = summarize_metric([], "faithfulness", expected_rows=0)
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=0)

    assert result["gate_passed"] is False
    assert result["reason"] == "empty_dataset"
    assert summary["mean"] is None


def test_fewer_valid_scores_fail_even_when_valid_mean_is_high():
    summary = summarize_metric(
        _rows([0.99, 0.99, 0.99, 0.99, None]),
        "faithfulness",
        expected_rows=5,
    )
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=5)

    assert summary["mean"] == 0.99
    assert summary["mean_scope"] == "valid_scores_only"
    assert summary["mean_denominator"] == 4
    assert result["gate_passed"] is False
    assert result["reason"] == "insufficient_valid_score_coverage"


def test_empty_contexts_are_invalid():
    rows = _rows([0.9])
    rows[0]["n_contexts"] = 0

    summary = summarize_metric(rows, "faithfulness", expected_rows=1)
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=1)

    assert result["gate_passed"] is False
    assert summary["invalid_ids"] == ["qa_001"]


def test_refused_answer_is_invalid():
    rows = _rows([0.9])
    rows[0]["refused"] = True

    summary = summarize_metric(rows, "faithfulness", expected_rows=1)
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=1)

    assert result["gate_passed"] is False
    assert summary["invalid_reasons"] == [{"id": "qa_001", "reason": "refused_answer"}]


def test_non_finite_baseline_scores_do_not_contribute_to_pass():
    baseline_rows = _rows([0.99] * 70 + [float("nan")] * 5)

    summary = summarize_metric(
        baseline_rows,
        "faithfulness",
        expected_rows=75,
    )
    result = evaluate_faithfulness_gate(summary, threshold=0.75, expected_rows=75)

    assert summary["n_valid"] == 70
    assert summary["n_invalid"] == 5
    assert math.isclose(summary["mean"], 0.99)
    assert result["gate_passed"] is False


def test_provider_or_evaluation_exception_maps_to_non_pass_without_message():
    exc = RuntimeError("request failed; sensitive provider detail")
    result = evaluation_failure_result(exc)

    assert result == {
        "gate_passed": False,
        "status": "infrastructure_error",
        "exit_code": 2,
        "error_type": "RuntimeError",
    }
    assert "sensitive provider detail" not in str(result)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold):
    summary = summarize_metric(_rows([0.9]), "faithfulness", expected_rows=1)

    with pytest.raises(ValueError, match="threshold"):
        evaluate_faithfulness_gate(summary, threshold=threshold, expected_rows=1)
