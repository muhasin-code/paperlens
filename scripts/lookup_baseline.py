#!/usr/bin/env python3
"""Dictionary lookup baseline for metrics/datasets."""

import json
import re
from collections import Counter


def build_vocab(train_path, min_count=2):
    metrics_counter = Counter()
    datasets_counter = Counter()
    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            out = json.loads(row["messages"][2]["content"])
            for m in out.get("metrics", []) or []:
                metrics_counter[m.lower()] += 1
            for d in out.get("datasets", []) or []:
                datasets_counter[d.lower()] += 1
    metrics_vocab = {m for m, c in metrics_counter.items() if c >= min_count}
    datasets_vocab = {d for d, c in datasets_counter.items() if c >= min_count}
    return metrics_vocab, datasets_vocab


def lookup_predict(text, vocab):
    found = []
    text_lower = text.lower()
    for term in vocab:
        if " " not in term:
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        else:
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.append(term)
    return found


def evaluate_lookup(val_path, metrics_vocab, datasets_vocab):
    tp = fp = fn = 0
    with open(val_path) as f:
        for line in f:
            row = json.loads(line)
            out = json.loads(row["messages"][2]["content"])
            # Extract chunk text from user message
            user_msg = row["messages"][1]["content"]
            if "Extract structured information from the following ML paper chunk:\n\n" in user_msg:
                chunk_text = user_msg.split(
                    "Extract structured information from the following ML paper chunk:\n\n"
                )[-1]
            else:
                chunk_text = user_msg

            for field, vocab in [("metrics", metrics_vocab), ("datasets", datasets_vocab)]:
                pred = lookup_predict(chunk_text, vocab)
                true = [x.lower() for x in (out.get(field) or [])]
                pred_set = set(pred)
                true_set = set(true)
                tp += len(pred_set & true_set)
                fp += len(pred_set - true_set)
                fn += len(true_set - pred_set)
    return tp, fp, fn


if __name__ == "__main__":
    metrics_vocab, datasets_vocab = build_vocab("data/extraction_dataset/train.jsonl")
    print(f"Metrics vocab: {len(metrics_vocab)} terms")
    print(f"Datasets vocab: {len(datasets_vocab)} terms")

    tp, fp, fn = evaluate_lookup("data/extraction_dataset/val.jsonl", metrics_vocab, datasets_vocab)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"Lookup baseline - P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # Per-field breakdown
    for field, vocab in [("metrics", metrics_vocab), ("datasets", datasets_vocab)]:
        tp_f = fp_f = fn_f = 0
        with open("data/extraction_dataset/val.jsonl") as f:
            for line in f:
                row = json.loads(line)
                out = json.loads(row["messages"][2]["content"])
                user_msg = row["messages"][1]["content"]
                chunk_text = user_msg.split(
                    "Extract structured information from the following ML paper chunk:\n\n"
                )[-1]
                pred = lookup_predict(chunk_text, vocab)
                true = [x.lower() for x in (out.get(field) or [])]
                pred_set = set(pred)
                true_set = set(true)
                tp_f += len(pred_set & true_set)
                fp_f += len(pred_set - true_set)
                fn_f += len(true_set - pred_set)
        p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 0
        r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) else 0
        f1_f = 2 * p_f * r_f / (p_f + r_f) if (p_f + r_f) else 0
        print(f"  {field}: P={p_f:.3f}, R={r_f:.3f}, F1={f1_f:.3f}")
