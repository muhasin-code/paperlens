#!/usr/bin/env python3
"""Assemble eval/gold_dataset.candidate.jsonl from LLM-drafted annotations.

Inputs (all gitignored, not in the repo):
  eval/gold_candidate_backfill.jsonl  LLM drafts for the 26 original queries
  eval/gold_candidate_new.jsonl       LLM drafts for the new rows
  eval/gold_qa_worksheet*.jsonl       chunk worksheets from prepare_gold_worksheet.py
Reads the frozen eval/gold_dataset.v2.5-retrieval.jsonl for the original 26 rows.
FORCE_SKIP, DROP and NEEDED_GENERATION are set for the Milestone 4.3 run.
Checks only ids, sentence counts and numbers. Spot-check answers against chunk text.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paperlens.evaluation.gold import DATASET_VERSION, load_gold_dataset
from src.paperlens.settings import get_settings

NOTES_NEW = "Annotated from chunks.parquet text of source_chunk_ids. No external sources."

# Back-fill rows kept retrieval-only even though ChatGPT answered them (weak answers).
FORCE_SKIP = {"qa_003", "qa_012", "qa_014", "qa_021"}

# Worksheet ids (the ids inside gold_candidate_new.jsonl) to exclude, e.g. {"qa_041", "qa_088"}.
DROP = set()

NEEDED_GENERATION = 81

frame = pd.read_parquet(get_settings().processed_chunks_path).set_index("chunk_id")


def jl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    ]


def check(ground_truth, chunk_ids, tag, warns):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", ground_truth.strip()) if s]
    if not 2 <= len(sentences) <= 6:
        warns.append(f"{tag}: {len(sentences)} sentences")
    if re.search(r"\[[^\]]*\]", ground_truth):
        warns.append(f"{tag}: bracket marker")
    for chunk_id in chunk_ids:
        if chunk_id not in frame.index:
            warns.append(f"{tag}: chunk id {chunk_id} not in chunks.parquet")
            return
    source = " ".join(frame.loc[c, "text"] for c in chunk_ids)
    for number in sorted(set(re.findall(r"\d+(?:\.\d+)?", ground_truth))):
        if number not in source:
            warns.append(f"{tag}: number {number} not in source text")


frozen = load_gold_dataset(Path("eval/gold_dataset.v2.5-retrieval.jsonl"))
back = {r["id"]: r for r in jl("eval/gold_candidate_backfill.jsonl")}
work = {}
for name in (
    "eval/gold_qa_worksheet.jsonl",
    "eval/gold_qa_worksheet_extra.jsonl",
    "eval/gold_qa_worksheet_extra2.jsonl",
):
    for row in jl(name):
        work[row["id"]] = row

new = [r for r in jl("eval/gold_candidate_new.jsonl") if r.get("status") == "ok"]
new = [r for r in new if r["id"] not in DROP]
print(
    f"{len(new)} new rows after drops (need {NEEDED_GENERATION - (len(frozen) - 10)} if 10 back-fill rows are retrieval-only)"
)

warns, rows, retrieval_only = [], [], []

# --- Back-fill: the 26 frozen rows, in file order ---
for i, old in enumerate(frozen, start=1):
    rid = f"qa_{i:03d}"
    b = back.get(rid)
    if b is None:
        sys.exit(f"{rid}: missing from eval/gold_candidate_backfill.jsonl")
    base = {"id": rid, "dataset_version": DATASET_VERSION, "query": old["query"]}
    if rid in FORCE_SKIP or b.get("status") != "ok":
        retrieval_only.append(rid)
        rows.append(
            {
                **base,
                "relevant_chunk_ids": old["relevant_chunk_ids"],
                "relevant_arxiv_ids": old["relevant_arxiv_ids"],
                "eval_sets": ["retrieval"],
                "notes": old["notes"],
            }
        )
        continue
    ids = b["source_chunk_ids"]
    missing = set(ids) - set(old["relevant_chunk_ids"])
    if missing:
        sys.exit(f"{rid}: source ids not in relevant_chunk_ids: {sorted(missing)}")
    check(b["ground_truth"], ids, rid, warns)
    rows.append(
        {
            **base,
            "ground_truth": b["ground_truth"],
            "relevant_chunk_ids": old["relevant_chunk_ids"],
            "relevant_arxiv_ids": old["relevant_arxiv_ids"],
            "source_chunk_ids": ids,
            "eval_sets": ["retrieval", "generation"],
            "notes": old["notes"],
        }
    )

# --- New rows: renumbered sequentially from qa_027 ---
seen_queries = {r["query"].lower() for r in rows}
seen_chunks = set()
for n, r in enumerate(new, start=len(frozen) + 1):
    tag = f"qa_{n:03d}"
    w = work.get(r["id"])
    if w is None:
        sys.exit(f"{r['id']}: not found in any worksheet file")
    if r["query"].lower() in seen_queries:
        warns.append(f"{tag} (worksheet {r['id']}): duplicate query")
    seen_queries.add(r["query"].lower())
    if w["chunk_id"] in seen_chunks:
        warns.append(f"{tag} (worksheet {r['id']}): chunk {w['chunk_id']} used twice")
    seen_chunks.add(w["chunk_id"])
    check(r["ground_truth"], [w["chunk_id"]], f"{tag} (worksheet {r['id']})", warns)
    rows.append(
        {
            "id": tag,
            "dataset_version": DATASET_VERSION,
            "query": r["query"],
            "ground_truth": r["ground_truth"],
            "relevant_chunk_ids": [w["chunk_id"]],
            "relevant_arxiv_ids": [w["arxiv_id"]],
            "source_chunk_ids": [w["chunk_id"]],
            "eval_sets": ["generation"],
            "notes": NOTES_NEW,
        }
    )

generation = sum(1 for r in rows if "generation" in r["eval_sets"])
if generation != NEEDED_GENERATION:
    warns.append(
        f"generation rows {generation} != {NEEDED_GENERATION} "
        f"({NEEDED_GENERATION - generation:+d} new rows needed)"
    )

out = Path("eval/gold_dataset.candidate.jsonl")
out.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    encoding="utf-8",
)
print(f"{len(rows)} rows, {generation} generation -> {out}")
print("retrieval-only:", ", ".join(retrieval_only) or "none")
print("\n".join(warns) if warns else "no warnings")
