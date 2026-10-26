"""Map gold rows to Ragas fields. Does not import ragas."""

from __future__ import annotations

from src.paperlens.evaluation.gold import is_generation_row


def generation_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if is_generation_row(row)]


def sample_fields(row: dict, response: str, contexts: list[str]) -> dict:
    return {
        "user_input": row["query"],
        "response": response or "",
        "retrieved_contexts": list(contexts),
        "reference": row["ground_truth"],
    }


def empty_context_scores(names: list[str]) -> dict[str, None]:
    return dict.fromkeys(names)


def mean_scores(rows: list[dict], names: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for name in names:
        values = [row[name] for row in rows if row.get(name) is not None]
        summary[name] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n": len(values),
        }
    return summary
