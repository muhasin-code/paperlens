from src.paperlens.evaluation.gold import validate_row


def test_generation_sources_can_be_outside_frozen_retrieval_labels():
    row = {
        "id": "qa_027",
        "dataset_version": "4.3.0",
        "query": "How does the method work?",
        "ground_truth": "The method uses the cited source.",
        "relevant_chunk_ids": ["2606.23188v1_chunk_0001"],
        "relevant_arxiv_ids": ["2606.23188v1"],
        "source_chunk_ids": ["2606.23188v1_chunk_0002"],
        "eval_sets": ["generation"],
    }

    assert validate_row(row, index=1) == []
