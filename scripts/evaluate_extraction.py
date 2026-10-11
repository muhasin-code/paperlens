#!/usr/bin/env python3
"""Batched evaluation for the PaperLens extraction adapter (QLoRA on Qwen2.5-3B-Instruct).

Usage:
    python scripts/evaluate_extraction.py [--limit N] [--output PATH] [--mode MODE] [--batch-size N]
    make evaluate-extraction

This script:
- Loads base model (Qwen/Qwen2.5-3B-Instruct) and fine-tuned adapter from HF Hub
- Loads test set from data/extraction_dataset/val.jsonl
- Runs batched inference on both models with the same prompts (chat template)
- Computes JSON validity rate, field-level F1, exact match, baselines, gold set
- Outputs results as JSON and prints formatted summary table
"""

import argparse
import gc
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PeftModel = None
    tqdm = None

from src.paperlens.extraction.annotator import DEFAULT_SYSTEM_PROMPT
from src.paperlens.settings import get_settings

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
FIELDS = ["research_question", "method", "datasets", "metrics", "key_finding"]
LIST_FIELDS = {"datasets", "metrics"}


def find_val_file(root: str = "/kaggle/input") -> Path:
    """Locate val.jsonl wherever the dataset got mounted."""
    matches = list(Path(root).rglob("val.jsonl"))
    if not matches:
        matches = list(Path(".").rglob("val.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No val.jsonl found under {root} or current directory")
    return matches[0]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(
    base_id: str = BASE_MODEL_ID,
    adapter_id: str | None = None,
    token: str | None = None,
    use_4bit: bool = True,
) -> tuple:
    """Load tokenizer and model (with optional LoRA adapter)."""
    tokenizer = AutoTokenizer.from_pretrained(base_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )

    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id, token=token)
    model.eval()
    return tokenizer, model


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def get_truth(ex: dict) -> dict | None:
    """Ground truth from `metadata` if present, else from trailing assistant message."""
    meta = ex.get("metadata") or {}
    if any(f in meta for f in FIELDS):
        return {f: meta.get(f) for f in FIELDS}
    msgs = ex.get("messages") or []
    if msgs and msgs[-1].get("role") == "assistant":
        parsed = parse_json(msgs[-1].get("content"))
        if parsed:
            return {f: parsed.get(f) for f in FIELDS}
    return None


def split_messages(ex: dict) -> tuple[str | None, str]:
    msgs = ex["messages"]
    system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
    user = next(m["content"] for m in msgs if m.get("role") == "user")
    return system, user


def build_prompt(tokenizer, system: str, user: str, mode: str, max_user_tokens: int) -> str:
    """Build prompt using chat template or raw format."""
    ids = tokenizer(user, add_special_tokens=False)["input_ids"]
    if len(ids) > max_user_tokens:
        user = tokenizer.decode(ids[:max_user_tokens])
    if mode == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if mode == "raw":
        return f"{system}\n\n{user}"
    raise ValueError(f"Unknown mode: {mode!r}")


@torch.inference_mode()
def generate_batched(
    tokenizer, model, prompts: list[str], batch_size: int = 8, max_new_tokens: int = 256
) -> list[str]:
    """Batched generation with length sorting and OOM back-off."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str | None] = [None] * len(prompts)
    bar = tqdm(total=len(prompts), desc="Generating")
    pos = 0
    while pos < len(order):
        idx = order[pos : pos + batch_size]
        try:
            enc = tokenizer([prompts[j] for j in idx], return_tensors="pt", padding=True).to(
                model.device
            )
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM -> batch_size={batch_size}")
            continue
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        for j, text in zip(
            idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True), strict=False
        ):
            outputs[j] = text
        pos += len(idx)
        bar.update(len(idx))
    bar.close()
    return outputs  # type: ignore


def _as_set(field: str, v) -> set[str]:
    if field in LIST_FIELDS:
        items = v if isinstance(v, list) else [v]
        return {str(x).strip().lower() for x in items}
    return set(str(v).lower().split())


def _prf(pred: set, truth: set) -> tuple[float, float, float]:
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_field(
    field: str, pred, truth, empty_is_null: bool = True
) -> tuple[tuple[float, float, float], bool]:
    """Returns ((precision, recall, f1), exact_match)."""
    if empty_is_null and field in LIST_FIELDS:
        pred = pred or None
        truth = truth or None
    if pred is None and truth is None:
        return (1.0, 1.0, 1.0), True
    if pred is None or truth is None:
        return (0.0, 0.0, 0.0), False
    a, b = _as_set(field, pred), _as_set(field, truth)
    return _prf(a, b), a == b if field in LIST_FIELDS else str(pred).strip().lower() == str(
        truth
    ).strip().lower()


def is_null(v) -> bool:
    return v is None or v == [] or v == ""


def lex_predict(vocab: dict, chunk: str, min_count: int = 2) -> list | None:
    u = chunk.lower()
    hits = [
        v
        for v, c in vocab.items()
        if c >= min_count and re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", u)
    ]
    return hits or None


def probe_prompt_modes(tokenizer, model, data: list[dict], n: int = 24, batch_size: int = 8) -> str:
    """Quick A/B on n examples to pick between chat-template and raw prompts."""
    scores = {}
    for mode in ("chat", "raw"):
        s = evaluate(tokenizer, model, data, mode=mode, limit=n, batch_size=batch_size)["summary"]
        mean_f1 = sum(v["f1"] for v in s["fields"].values()) / len(FIELDS)
        scores[mode] = (s["json_valid_rate"], mean_f1)
        print(f"{mode:>4}: valid={s['json_valid_rate']:.0%}  mean F1={mean_f1:.3f}")
    best = max(scores, key=lambda m: scores[m])
    print(f"-> using '{best}'")
    return best


def evaluate(
    tokenizer,
    model,
    data: list[dict],
    mode: str = "chat",
    batch_size: int = 8,
    max_new_tokens: int = 256,
    max_user_tokens: int = 900,
    limit: int | None = None,
    empty_is_null: bool = True,
    adapter_repo: str | None = None,
    label: str = "model",
) -> dict:
    """Run evaluation on a dataset."""
    rows = data[:limit] if limit else data

    prompts, truths = [], []
    for i, ex in enumerate(rows):
        truth = get_truth(ex)
        if truth is None:
            raise ValueError(f"Row {i}: no ground truth found. Keys present: {list(ex.keys())}")
        system, user = split_messages(ex)
        prompts.append(
            build_prompt(tokenizer, system or DEFAULT_SYSTEM_PROMPT, user, mode, max_user_tokens)
        )
        truths.append(truth)

    t0 = time.time()
    raw = generate_batched(tokenizer, model, prompts, batch_size, max_new_tokens)
    elapsed = time.time() - t0

    acc = {f: {"precision": [], "recall": [], "f1": [], "exact": []} for f in FIELDS}
    details, valid = [], 0
    for i, (text, truth) in enumerate(zip(raw, truths, strict=False)):
        pred = parse_json(text)
        is_valid = pred is not None and all(f in pred for f in FIELDS)
        valid += is_valid
        for f in FIELDS:
            if pred is None:
                (p, r, f1), ex_ok = (0.0, 0.0, 0.0), False
            else:
                (p, r, f1), ex_ok = score_field(f, pred.get(f), truth[f], empty_is_null)
            for k, v in zip(
                ("precision", "recall", "f1", "exact"), (p, r, f1, ex_ok), strict=False
            ):
                acc[f][k].append(float(v))
        details.append(
            {
                "index": i,
                "valid_json": is_valid,
                "raw_output": text,
                "predicted": pred,
                "ground_truth": truth,
            }
        )

    n = len(rows)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL_ID,
        "adapter_repo": adapter_repo,
        "label": label,
        "prompt_mode": mode,
        "total_examples": n,
        "json_valid_rate": valid / n if n else 0.0,
        "seconds": round(elapsed, 1),
        "sec_per_example": round(elapsed / n, 2) if n else None,
        "fields": {f: {k: mean(v) for k, v in acc[f].items()} for f in FIELDS},
        "scoring": "macro-averaged per-example set-F1 (token set for text fields, item set for list fields)",
        "empty_is_null": empty_is_null,
    }
    return {"summary": summary, "examples": details}


def print_report(res: dict) -> None:
    s = res["summary"]
    print("=" * 64)
    print(
        f"EVALUATION [{s.get('label', 'model')}]  ({s['total_examples']} examples, prompt_mode={s['prompt_mode']})"
    )
    print(
        f"JSON validity: {s['json_valid_rate']:.1%}   time: {s['seconds']}s ({s['sec_per_example']} s/ex)"
    )
    print("-" * 64)
    print(f"{'field':20s} {'P':>6} {'R':>6} {'F1':>6} {'exact':>7}")
    for f, v in s["fields"].items():
        print(f"{f:20s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['exact']:7.1%}")


def save_results(res: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


def run_baselines(
    res: dict, base_res: dict, val_data: list[dict], train_data: list[dict]
) -> tuple[dict, dict]:
    """Compute all-null and dictionary lookup baselines."""
    from collections import Counter

    allnull = {
        f: mean(score_field(f, None, e["ground_truth"][f])[0][2] for e in res["examples"])
        for f in FIELDS
    }
    lookup = {}
    tr_t = [get_truth(e) for e in train_data]
    for f in ("metrics", "datasets"):
        vocab = Counter(x for t in tr_t for x in (t[f] or []))
        lookup[f] = mean(
            score_field(f, lex_predict(vocab, split_messages(e)[1]), get_truth(e)[f])[0][2]
            for e in val_data
        )
        print(f"{f}: lookup vocabulary = {len(vocab)} names from train labels")
    return allnull, lookup


def span_breakdown(r: dict, label: str) -> None:
    """Analyze null decision vs span quality for method/key_finding."""
    for f in ("method", "key_finding"):
        nn = [e for e in r["examples"] if e["predicted"] and not is_null(e["ground_truth"][f])]
        nl = [e for e in r["examples"] if e["predicted"] and is_null(e["ground_truth"][f])]
        f1_nn = (
            mean(score_field(f, e["predicted"].get(f), e["ground_truth"][f])[0][2] for e in nn)
            if nn
            else float("nan")
        )
        null_ok = mean(is_null(e["predicted"].get(f)) for e in nl) if nl else float("nan")
        print(
            f"[{label:4s}] {f:12s} truth non-null (n={len(nn)}): mean F1 {f1_nn:.3f} | truth null (n={len(nl)}): predicts null {null_ok:.0%}"
        )


def gold_eval(r: dict, val_data: list[dict], gold: dict) -> dict:
    """Evaluate against gold set for method/key_finding."""
    out = {}
    for f, gk in (("method", "gold_method"), ("key_finding", "gold_key_finding")):
        f1s, pred_null = [], 0
        for ex in r["examples"]:
            g = gold.get(val_data[ex["index"]]["metadata"]["chunk_id"])
            if not g or is_null(g.get(gk)):
                continue
            pred = (ex["predicted"] or {}).get(f)
            pred_null += is_null(pred)
            f1s.append(score_field(f, pred, g[gk])[0][2])
        out[f] = {"n": len(f1s), "f1": mean(f1s) if f1s else None, "predicted_null": pred_null}
    return out


def print_criteria(
    res: dict, base_res: dict, allnull: dict, lookup: dict, gold_scores: dict | None
) -> list:
    """Print and return pass/fail criteria."""
    checks = []

    def chk(name, ok, detail):
        checks.append((name, "PASS" if ok else "FAIL", detail))

    def get_f1(r, f):
        return r["summary"]["fields"][f]["f1"]

    v3s = res["summary"]
    chk("JSON validity > 95%", v3s["json_valid_rate"] > 0.95, f"{v3s['json_valid_rate']:.1%}")
    chk(
        "method F1 > 0.80 and > base",
        get_f1(res, "method") > 0.80 and get_f1(res, "method") > get_f1(base_res, "method"),
        f"v3 {get_f1(res, 'method'):.3f} | base {get_f1(base_res, 'method'):.3f}",
    )
    chk(
        "datasets F1 > lookup and > base",
        get_f1(res, "datasets") > lookup["datasets"]
        and get_f1(res, "datasets") > get_f1(base_res, "datasets"),
        f"v3 {get_f1(res, 'datasets'):.3f} | lookup {lookup['datasets']:.3f} | base {get_f1(base_res, 'datasets'):.3f}",
    )
    chk(
        "key_finding F1 > 0.60, > base and > all-null",
        get_f1(res, "key_finding") > 0.60
        and get_f1(res, "key_finding") > get_f1(base_res, "key_finding")
        and get_f1(res, "key_finding") > allnull["key_finding"],
        f"v3 {get_f1(res, 'key_finding'):.3f} | base {get_f1(base_res, 'key_finding'):.3f} | all-null {allnull['key_finding']:.3f}",
    )
    chk(
        "metrics F1 > lookup and > base",
        get_f1(res, "metrics") > lookup["metrics"]
        and get_f1(res, "metrics") > get_f1(base_res, "metrics"),
        f"v3 {get_f1(res, 'metrics'):.3f} | lookup {lookup['metrics']:.3f} | base {get_f1(base_res, 'metrics'):.3f}",
    )

    if gold_scores:
        for f, thr in (("method", 0.70), ("key_finding", 0.50)):
            s = gold_scores["v3"][f]
            if s and s["n"] and s["f1"] is not None:
                chk(f"gold {f} F1 > {thr:.2f}", s["f1"] > thr, f"{s['f1']:.3f} on n={s['n']}")
            else:
                checks.append((f"gold {f} F1 > {thr:.2f}", "SKIPPED", "no gold scores available"))

    print(f"{'criterion':46s} {'result':8s} detail")
    for name, status, detail in checks:
        print(f"{name:46s} {status:8s} {detail}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 50 examples only")
    parser.add_argument("--mode", choices=["chat", "raw"], default="chat", help="Prompt mode")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument(
        "--gold-path", type=Path, default=None, help="Path to gold_extraction.jsonl"
    )
    args = parser.parse_args()

    settings = get_settings()
    adapter_repo = settings.extraction_adapter_repo or settings.hf_hub_model_id

    val_path = find_val_file()
    val_data = load_jsonl(val_path)
    train_data = load_jsonl(val_path.parent / "train.jsonl")

    print(f"Loaded {len(val_data)} val / {len(train_data)} train examples from {val_path}")

    # Load gold set if provided
    gold = None
    if args.gold_path and args.gold_path.exists():
        gold_rows = load_jsonl(args.gold_path)
        gold = {r["chunk_id"]: r for r in gold_rows}
        print(f"Loaded gold set: {len(gold)} entries")

    # Load model
    hf_token = settings.hf_token or None
    tokenizer, model = load_model(
        adapter_id=adapter_repo, token=hf_token, use_4bit=not args.no_4bit
    )
    print(f"Adapter: {adapter_repo}")

    # Probe prompt mode
    mode = args.mode
    if mode == "chat":
        mode = probe_prompt_modes(tokenizer, model, val_data, n=24, batch_size=8)

    # Dry run
    if args.dry_run:
        dry = evaluate(tokenizer, model, val_data, mode=mode, limit=50, batch_size=16, label="dry")
        print_report(dry)
        bad = [e for e in dry["examples"] if not e["valid_json"]]
        print(f"\nInvalid outputs: {len(bad)}")
        for e in bad[:5]:
            n_tok = len(tokenizer(e["raw_output"], add_special_tokens=False)["input_ids"])
            print(f"[{e['index']}] ~{n_tok} new tokens (cap 256)\n...{e['raw_output'][-300:]}\n")
        return 0

    # Full evaluation: fine-tuned
    t0 = time.time()
    res = evaluate(tokenizer, model, val_data, mode=mode, batch_size=args.batch_size, label="v3")
    print(f"Wall time: {time.time() - t0:.0f}s")
    print_report(res)

    # Full evaluation: base (adapter disabled)
    t0 = time.time()
    with model.disable_adapter():
        base_res = evaluate(
            tokenizer,
            model,
            val_data,
            mode=mode,
            batch_size=args.batch_size,
            adapter_repo=None,
            label="base",
        )
    print(f"Wall time: {time.time() - t0:.0f}s")
    print_report(base_res)

    # Baselines
    allnull, lookup = run_baselines(res, base_res, val_data, train_data)

    # Span breakdown
    span_breakdown(res, "v3")
    span_breakdown(base_res, "base")

    # Gold set evaluation
    gold_scores = None
    if gold:
        gold_scores = {
            "v3": gold_eval(res, val_data, gold),
            "base": gold_eval(base_res, val_data, gold),
        }
        n_in_val = sum(g in {e["metadata"]["chunk_id"] for e in val_data} for g in gold)
        print(f"gold chunks present in val: {n_in_val} of {len(gold)}")
        for f in ("method", "key_finding"):
            for name in ("v3", "base"):
                s = gold_scores[name][f]
                f1 = "n/a" if s["f1"] is None else f"{s['f1']:.3f}"
                print(
                    f"[{name:4s}] gold {f:12s} n={s['n']:3d}  F1={f1}  predicted null on {s['predicted_null']} of them"
                )

    # Success criteria
    checks = print_criteria(res, base_res, allnull, lookup, gold_scores)

    # Save summary
    summary = {
        "v3": res["summary"],
        "base": base_res["summary"],
        "all_null_f1": allnull,
        "lookup_f1": lookup,
        "gold": gold_scores,
        "criteria": [{"name": n, "status": s, "detail": d} for n, s, d in checks],
    }
    out_path = args.output or Path("extraction_eval_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
