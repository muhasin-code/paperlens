#!/usr/bin/env python3
"""Sample corpus chunks into a worksheet. Does not write the gold file."""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.evaluation.gold import load_gold_dataset
from src.paperlens.settings import get_settings


def _existing_ids(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    chunk_ids: set[str] = set()
    arxiv_ids: set[str] = set()
    for row in load_gold_dataset(path):
        for chunk_id in row.get("relevant_chunk_ids") or []:
            chunk_ids.add(chunk_id)
        for arxiv_id in row.get("relevant_arxiv_ids") or []:
            arxiv_ids.add(arxiv_id)
    return chunk_ids, arxiv_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft generation-gold worksheet")
    parser.add_argument("--n", type=int, default=49)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--start-id", type=int, default=27)
    parser.add_argument("--out", type=Path, default=Path("eval/gold_qa_worksheet.jsonl"))
    args = parser.parse_args(argv)

    settings = get_settings()
    frame = pd.read_parquet(settings.processed_chunks_path)
    if "token_count" in frame.columns:
        frame = frame[frame["token_count"] >= 80]
    if "section_label" in frame.columns:
        labels = frame["section_label"].fillna("").str.lower()
        frame = frame[~labels.str.contains("reference|acknowledg", regex=True)]

    used_chunks, used_papers = _existing_ids(Path(settings.gold_dataset_path))
    frame = frame[~frame["chunk_id"].isin(used_chunks)]

    fresh = frame[~frame["arxiv_id"].isin(used_papers)]
    pool = fresh if len(fresh["arxiv_id"].unique()) >= args.n else frame
    pool = pool.drop_duplicates(subset="arxiv_id")
    if len(pool) < args.n:
        print(f"Only {len(pool)} eligible papers, requested {args.n}", file=sys.stderr)
        return 1

    picked = pool.sample(n=args.n, random_state=random.Random(args.seed).randint(0, 2**31 - 1))
    # pandas sample random_state must be an int. Use the seed directly.
    picked = pool.sample(n=args.n, random_state=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for offset, row in enumerate(picked.itertuples(index=False)):
            item = {
                "id": f"qa_{args.start_id + offset:03d}",
                "chunk_id": row.chunk_id,
                "arxiv_id": row.arxiv_id,
                "title": row.title,
                "section_label": row.section_label,
                "text": row.text,
                "query": "",
                "ground_truth": "",
                "status": "draft",
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} drafts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
