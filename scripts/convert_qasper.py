#!/usr/bin/env python3
"""Convert QASPER dataset to PaperLens extraction SFT format."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.extraction.models import ExtractionExample, ExtractionOutput

DEFAULT_SYSTEM_PROMPT = """You are a research assistant specialized in extracting structured information from machine learning papers.

From the provided paper chunk, extract the following fields and return ONLY a JSON object with these keys (all nullable):

- research_question: The core problem or hypothesis the paper addresses
- method: The proposed approach or algorithm (max one paragraph)
- datasets: List of dataset names used (empty array if none)
- metrics: List of evaluation metrics reported (empty array if none)
- key_finding: The main result or contribution in one sentence

If a field is not present or not useful, set it to null. Do not hallucinate information. Do not add any prose - only return the JSON object.

Example output:
{"research_question": "How to improve...", "method": "...", "datasets": ["..."], "metrics": ["..."], "key_finding": "..."}"""


EXTRACTABLE_SECTIONS = ["method", "experiments", "results", "evaluation"]

CITED_DATASETS = [
    "ImageNet",
    "CIFAR-10",
    "CIFAR-100",
    "MNIST",
    "Fashion-MNIST",
    "SVHN",
    "Coco",
    "COCO",
    "VOC",
    "ImageNet-1K",
    "ImageNet-100",
    "ImageNet-R",
    "ImageNet-A",
    "Visual Genome",
    "CLEVR",
    "T5",
    "GLUE",
    "SuperGLUE",
    "SQuAD",
    "HotpotQA",
    "TriviaQA",
    "WebOfQuestions",
    "CurriculumNarrativeQA",
    "NarrativeQA",
    "BERT",
    "GPT",
    "XLNet",
    "RoBERTa",
    "ELECTRA",
    "DistilBERT",
    "ALBERT",
    "DeBERTa",
    "PaLM",
    "Chinchilla",
    "Mistral",
    "Phi",
    "Gemma",
    "Qwen",
    "Claude",
    "Gemini",
]

CITED_METRICS = [
    "accuracy",
    "acc",
    "accuracy@1",
    "top-1 accuracy",
    "top-5 accuracy",
    "F1",
    "F1 score",
    "BLEU",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "ROUGE",
    "ROUGE-L",
    "ROUGE-1",
    "ROUGE-2",
    "METEOR",
    "CIDER",
    "SPICE",
    "BERTScore",
    "MOVERScore",
    "precision",
    "recall",
    "AP",
    "AR",
    "mAP",
    "IoU",
    "Dice",
    "Jaccard",
    "loss",
    "perplexity",
    "ppl",
    "efficiency",
    "latency",
    "AUROC",
    "AUPRC",
    "FPR",
    "FNR",
    "AUC",
]


def extract_section_text(full_text: str, section_names: list[str]) -> dict[str, str]:
    sections = {}
    for target in section_names:
        # Simple case-insensitive search for section header
        pattern = rf"(?i){re.escape(target)}[:\s]\n?(.*?)(?=(?:\n[A-Z][a-z]+:)|$)"
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            if len(text) >= 100:
                sections[target] = text[:2000]
    return sections


def find_datasets_in_text(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for dataset in CITED_DATASETS:
        if dataset.lower() in text_lower:
            found.append(dataset)
    return list(set(found))


def find_metrics_in_text(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for metric in CITED_METRICS:
        if metric.lower() in text_lower:
            found.append(metric)
    return list(set(found))


def build_qas_list(qas_dict: dict) -> list[dict]:
    """Convert QASPER qas dict (parallel lists) to list of QA dicts."""
    n = len(qas_dict.get("question", []))
    result = []
    for i in range(n):
        result.append(
            {
                "question": qas_dict.get("question", [""])[i]
                if i < len(qas_dict.get("question", []))
                else "",
                "answers": qas_dict.get("answers", [{}])[i]
                if i < len(qas_dict.get("answers", []))
                else {},
            }
        )
    return result


def extract_answer_text(answers: dict) -> str | None:
    """Extract answer text from QASPER answers dict."""
    if not isinstance(answers, dict):
        return None
    answer_list = answers.get("answer", [])
    for ans in answer_list:
        if isinstance(ans, dict):
            # Use free_form_answer field
            text = ans.get("free_form_answer", "")
        else:
            text = str(ans)
        if text and text.strip():
            return text.strip()
    return None


def generate_extraction_example(
    paper: dict,
    chunk_text: str,
    section_label: str,
    system_prompt: str,
) -> ExtractionExample | None:
    qas_dict = paper.get("qas", {})
    qas_list = build_qas_list(qas_dict)

    research_question = None
    key_finding = None

    for qa in qas_list:
        question = qa.get("question", "")
        answers = qa.get("answers", {})

        if not research_question and question:
            research_question = question

        ans_text = extract_answer_text(answers)
        if ans_text and not key_finding:
            key_finding = ans_text
            if len(key_finding) > 500:
                key_finding = key_finding[:500]

        if research_question and key_finding:
            break

    if not research_question:
        research_question = "Extract structured information from this ML paper section."

    method_text = None
    datasets_list = find_datasets_in_text(chunk_text)
    metrics_list = find_metrics_in_text(chunk_text)

    if section_label == "method":
        method_text = chunk_text[:500] if chunk_text else None

    output = ExtractionOutput(
        research_question=research_question,
        method=method_text,
        datasets=datasets_list if datasets_list else None,
        metrics=metrics_list if metrics_list else None,
        key_finding=key_finding,
    )

    return ExtractionExample(
        chunk_id=f"{paper.get('id', 'unknown')}_{section_label}",
        arxiv_id=paper.get("id", "unknown"),
        section_label=section_label,
        input_text=chunk_text,
        output=output,
        system_prompt=system_prompt,
    )


def convert_qasper(
    limit: int = 2000,
    output_dir: Path = Path("./data/extraction_dataset"),
    val_split: float = 0.1,
    dry_run: bool = False,
) -> dict:
    from datasets import load_dataset

    print("Loading allenai/qasper dataset...")
    try:
        dataset = load_dataset("allenai/qasper", split="train")
    except Exception as e:
        print(f"ERROR: Failed to load QASPER dataset: {e}")
        return {"total_examples": 0, "train_size": 0, "val_size": 0, "dry_run": dry_run}

    print(f"Loaded {len(dataset)} papers from QASPER")

    if limit and len(dataset) > limit:
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)
        print(f"Sampled {len(dataset)} papers (limit={limit})")

    examples: list[ExtractionExample] = []

    skipped_no_full_text = 0
    skipped_no_answer = 0

    for i, paper in enumerate(dataset):
        if i % 100 == 0 and i > 0:
            print(f"Processing paper {i+1}/{len(dataset)}...")

        full_text = paper.get("full_text")
        if not full_text:
            skipped_no_full_text += 1
            continue

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        qas_dict = paper.get("qas", {})
        qas_list = build_qas_list(qas_dict)
        answered_qas = [
            qa
            for qa in qas_list
            if qa.get("answers", {}).get("answer", [""])[0]
            if qa.get("answers", {}).get("answer")
        ]

        if not answered_qas:
            skipped_no_answer += 1
            continue

        full_paper_text = f"Title: {title}\n\nAbstract: {abstract}\n\n"
        if isinstance(full_text, dict):
            section_names = full_text.get("section_name", [])
            paragraphs_list = full_text.get("paragraphs", [])
            for sn, para in zip(section_names, paragraphs_list, strict=False):
                full_paper_text += f"{sn}: {para}\n\n"
        elif isinstance(full_text, list):
            for section in full_text:
                if isinstance(section, dict):
                    section_name = section.get("section_name", "")
                    paragraphs = section.get("paragraphs", "")
                    full_paper_text += f"{section_name}: {paragraphs}\n\n"
                else:
                    full_paper_text += f"{section}\n\n"
        elif isinstance(full_text, str):
            full_paper_text += full_text

        for section_name in EXTRACTABLE_SECTIONS:
            section_text = extract_section_text(full_paper_text, [section_name]).get(section_name)
            if not section_text or len(section_text.strip()) < 100:
                continue

            example = generate_extraction_example(
                paper=paper,
                chunk_text=section_text,
                section_label=section_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )

            if example:
                jsonl_line = json.dumps(example.to_sft_row())
                try:
                    parsed = json.loads(jsonl_line)
                    if parsed.get("messages"):
                        examples.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Generated invalid JSONL for {example.chunk_id}")

    print("\nConversion complete:")
    print(f"  Total examples generated: {len(examples)}")
    print(f"  Skipped (no answered questions): {skipped_no_answer}")
    print(f"  Skipped (no full_text): {skipped_no_full_text}")

    if len(examples) < 30:
        print("ERROR: Generated too few examples. Check the QASPER schema and extraction logic.")
        return {
            "total_examples": 0,
            "train_size": 0,
            "val_size": 0,
            "dry_run": dry_run,
        }

    random.shuffle(examples)
    val_size = max(1, int(len(examples) * val_split))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_sft_row()) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex.to_sft_row()) + "\n")

        print("\nWrote files:")
        print(f"  {train_path}")
        print(f"  {val_path}")

        if train_path.exists():
            train_size = train_path.stat().st_size / 1024
            print(f"  train.jsonl size: {train_size:.1f} KB ({len(train_examples)} examples)")
        if val_path.exists():
            val_size_kb = val_path.stat().st_size / 1024
            print(f"  val.jsonl size: {val_size_kb:.1f} KB ({len(val_examples)} examples)")

    return {
        "total_examples": len(examples),
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert QASPER dataset to PaperLens extraction format"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of papers to process (default: 2000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/extraction_dataset"),
        help="Output directory for train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show statistics without writing files"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QASPER to PaperLens Extraction Dataset Converter")
    print("=" * 60)
    print(f"Limit: {args.limit}")
    print(f"Output dir: {args.output_dir}")
    print(f"Val split: {args.val_split}")
    print(f"Dry run: {args.dry_run}")
    print()

    result = convert_qasper(
        limit=args.limit,
        output_dir=args.output_dir,
        val_split=args.val_split,
        dry_run=args.dry_run,
    )

    return 0 if result["total_examples"] >= 30 else 1


if __name__ == "__main__":
    sys.exit(main())
