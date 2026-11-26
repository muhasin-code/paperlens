import json

import pytest
from scripts.evaluate_ragas import _has_complete_scores, _load_or_create_checkpoint


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
