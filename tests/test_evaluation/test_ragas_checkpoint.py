import json
from pathlib import Path

import pytest
from scripts.evaluate_ragas import (
    _has_complete_scores,
    _load_or_create_checkpoint,
    _parse_args,
    _summarize_results,
)


def _scored_rows(scores: list[object]) -> list[dict]:
    return [
        {
            "id": f"qa_{index + 1:03}",
            "n_contexts": 2,
            "refused": False,
            "faithfulness": score,
        }
        for index, score in enumerate(scores)
    ]


def test_fail_under_parse_selects_ci_result_path():
    args = _parse_args(["--fail-under", "0.75"])
    assert args.fail_under == 0.75
    assert args.output == Path("eval/ragas_ci_result.json")


def test_invalid_fail_under_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--fail-under", "nan"])
    assert exc_info.value.code == 2


def test_baseline_mode_does_not_enforce_gate():
    args = _parse_args([])
    assert args.fail_under is None
    assert args.output == Path("eval/ragas_baseline.json")

    rows, summaries, gate, exit_code = _summarize_results(
        _scored_rows([0.75]),
        ["faithfulness"],
        expected_rows=1,
        threshold=args.fail_under,
    )

    assert rows[0]["faithfulness"] == 0.75
    assert summaries["faithfulness"]["n_valid"] == 1
    assert gate is None
    assert exit_code == 0


@pytest.mark.parametrize(
    ("scores", "expected_code", "expected_reason"),
    [
        ([0.74, 0.74], 1, "below_threshold"),
        ([0.75, 0.75], 0, "passed"),
        ([0.8, 0.9], 0, "passed"),
        ([0.99, None], 1, "insufficient_valid_score_coverage"),
        ([0.99, float("nan")], 1, "insufficient_valid_score_coverage"),
    ],
)
def test_gate_result_and_exit_code_without_provider_calls(
    scores,
    expected_code,
    expected_reason,
):
    rows, summaries, gate, exit_code = _summarize_results(
        _scored_rows(scores),
        ["faithfulness"],
        expected_rows=len(scores),
        threshold=0.75,
    )

    assert exit_code == expected_code
    assert gate["reason"] == expected_reason

    if expected_reason == "insufficient_valid_score_coverage":
        assert summaries["faithfulness"]["n_valid"] == len(scores) - 1
        assert rows[-1]["faithfulness"] is None


@pytest.mark.parametrize(
    "score",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        -0.1,
        1.1,
    ],
)
def test_checkpoint_with_invalid_score_is_not_complete(score):
    assert not _has_complete_scores(
        {"faithfulness": score},
        ["faithfulness"],
    )


def test_checkpoint_with_finite_score_is_complete():
    assert _has_complete_scores(
        {"faithfulness": 0.75},
        ["faithfulness"],
    )


def test_checkpoint_can_be_resumed_with_matching_metadata(tmp_path):
    path = tmp_path / "baseline.checkpoint.jsonl"
    metadata = {"row_ids": ["qa_001"], "retrieval_top_k": 5}
    _load_or_create_checkpoint(path, metadata, resume=False)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "qa_001", "faithfulness": 0.8}) + "\n")

    assert _load_or_create_checkpoint(path, metadata, resume=True) == {
        "qa_001": {"id": "qa_001", "faithfulness": 0.8}
    }


def test_checkpoint_rejects_mismatched_metadata(tmp_path):
    path = tmp_path / "baseline.checkpoint.jsonl"
    _load_or_create_checkpoint(path, {"retrieval_top_k": 5}, resume=False)

    with pytest.raises(ValueError, match="metadata does not match"):
        _load_or_create_checkpoint(path, {"retrieval_top_k": 1}, resume=True)


def test_only_rows_with_all_scores_are_skipped_on_resume():
    assert _has_complete_scores(
        {"faithfulness": 0.0, "answer_relevancy": 0.5}, ["faithfulness", "answer_relevancy"]
    )
    assert not _has_complete_scores(
        {"faithfulness": 0.8, "answer_relevancy": None}, ["faithfulness", "answer_relevancy"]
    )
