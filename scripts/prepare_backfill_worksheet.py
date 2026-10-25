#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paperlens.evaluation.gold import load_gold_dataset
from src.paperlens.settings import get_settings

settings = get_settings()
frame = pd.read_parquet(settings.processed_chunks_path).set_index("chunk_id")
rows = load_gold_dataset(Path("eval/gold_dataset.v2.5-retrieval.jsonl"))
out = Path("eval/gold_backfill_worksheet.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, row in enumerate(rows, start=1):
        chunks = [{"chunk_id": c, "text": frame.loc[c, "text"]} for c in row["relevant_chunk_ids"]]
        f.write(
            json.dumps(
                {"id": f"qa_{i:03d}", "query": row["query"], "chunks": chunks}, ensure_ascii=False
            )
            + "\n"
        )
print(f"Wrote {len(rows)} rows to {out}")
