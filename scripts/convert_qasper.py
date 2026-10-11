#!/usr/bin/env python3
"""Convert QASPER dataset to PaperLens extraction SFT format.

This script loads the allenai/qasper dataset from HuggingFace and converts it to the
ExtractionExample SFT format used for QLoRA fine-tuning. It extracts sections matching
'method', 'experiments', 'results', 'evaluation' from the full text and generates
structured extraction examples WHERE LABELS ARE DERIVED FROM THE CHUNK TEXT to match
the system prompt: 'Extract structured information from the following ML paper chunk'.

Usage:
    python scripts/convert_qasper.py [--limit N] [--output-dir PATH] [--val-split FRAC] [--dry-run]
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
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

# Metrics with word-boundary-safe matching (removed "acc", "F1" duplicate, "AP" kept but careful)
CITED_METRICS = [
    "accuracy",
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
    """Extract text for specific sections from the QASPER full_text."""
    sections = {}
    for target in section_names:
        pattern = rf"(?i){re.escape(target)}[:\s]\n?(.*?)(?=(?:\n[A-Z][a-z]+:)|$)"
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            if len(text) >= 100:
                sections[target] = text[:2000]
    return sections


def find_datasets_in_text(text: str) -> list[str]:
    """Find dataset mentions in text (case-insensitive substring)."""
    found = []
    text_lower = text.lower()
    for dataset in CITED_DATASETS:
        if dataset.lower() in text_lower:
            found.append(dataset)
    return list(set(found))


def find_metrics_in_text(text: str) -> list[str]:
    """Find metric mentions using word boundaries for single tokens, phrase for multi-word.

    Only matches metrics that appear as standalone tokens to avoid false positives
    (e.g., "AR" in "area", "are" should not match).
    """
    found = []
    text_lower = text.lower()
    for metric in CITED_METRICS:
        metric_lower = metric.lower()
        if " " not in metric_lower:
            # Single token: require word boundaries AND not part of longer word
            # Use negative lookahead/lookbehind for alphanumeric
            pattern = rf"(?<![a-z0-9]){re.escape(metric_lower)}(?![a-z0-9])"
            if re.search(pattern, text_lower):
                found.append(metric)
        else:
            # Multi-word: phrase search with word boundaries at edges
            pattern = rf"(?<![a-z0-9]){re.escape(metric_lower)}(?![a-z0-9])"
            if re.search(pattern, text_lower):
                found.append(metric)
    return list(set(found))


def infer_research_question(title: str, abstract: str, section_label: str, chunk_text: str) -> str:
    """Infer a research question from paper metadata and section."""
    section_questions = {
        "method": f"What method does the paper '{title}' propose?",
        "experiments": f"What experimental setup does the paper '{title}' use?",
        "results": f"What are the main results of the paper '{title}'?",
        "evaluation": f"How does the paper '{title}' evaluate its method?",
    }

    base_q = section_questions.get(section_label, f"What does the paper '{title}' address?")

    if abstract:
        first_sentence = abstract.split(".")[0].strip()
        if len(first_sentence) > 20:
            return f"{base_q} Context: {first_sentence}"

    return base_q


def extract_key_finding_from_chunk(chunk_text: str) -> str | None:
    """Extract a key finding from results/evaluation chunk text."""
    if not chunk_text or len(chunk_text.strip()) < 50:
        return None

    finding_patterns = [
        r"(?:we (?:find|show|demonstrate|observe|achieve|report|see|get|reach))\s+([^.]{20,200})",
        r"(?:our (?:method|model|approach|results?))\s+(?:achieves?|shows?|reaches?|gets?|obtains?)\s+([^.]{20,200})",
        r"(?:the (?:results?|findings?|performance|accuracy|score))\s+(?:shows?|indicate?|suggest?|is|are)\s+([^.]{20,200})",
        r"(?:outperform|improv|beat|surpass|exceed)\s+([^.]{20,200})",
        r"(?:achieve[sd]?|reach[ed]?|obtain[ed]?)\s+(?:an?\s+)?([\d.]+(?:\s*[%%]|\s+points?))",
    ]

    for pattern in finding_patterns:
        match = re.search(pattern, chunk_text, re.IGNORECASE)
        if match:
            finding = match.group(1).strip()
            if len(finding) > 15:
                return finding[:500]

    # Fallback: first substantial sentence with numbers/metrics
    sentences = re.split(r"[.!?]", chunk_text)
    for sent in sentences:
        sent = sent.strip()
        if (
            len(sent) > 30
            and any(c.isdigit() for c in sent)
            and any(
                w in sent.lower()
                for w in [
                    "acc",
                    "f1",
                    "score",
                    "performance",
                    "result",
                    "improv",
                    "achiev",
                    "outperform",
                    "beat",
                ]
            )
        ):
            return sent[:500]

    return None


def generate_extraction_example(
    paper: dict,
    chunk_text: str,
    section_label: str,
    system_prompt: str,
    paper_idx: int,
) -> ExtractionExample | None:
    """Generate an ExtractionExample from QASPER paper and section text."""

    # Get paper ID - try multiple fields, fallback to index
    paper_id = (
        paper.get("paper_id")
        or paper.get("id")
        or paper.get("paper_id_str")
        or f"paper_{paper_idx}"
    )

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Build user prompt with metadata(so research_question is answerable)
    user_content = (
        f"Paper Title: {title}\n\n"
        f"Paper Abstract: {abstract}\n\n"
        f"Section: {section_label}\n\n"
        f"Extract structured information from the ML paper chunk:\n\n{chunk_text}"
    )

    research_question = infer_research_question(title, abstract, section_label, chunk_text)

    method_text = None
    if section_label == "method":
        method_text = chunk_text[:500] if chunk_text else None

    datasets_list = find_datasets_in_text(chunk_text)
    metrics_list = find_metrics_in_text(chunk_text)

    key_finding = None
    if section_label in ("results", "evaluation"):
        key_finding = extract_key_finding_from_chunk(chunk_text)

    output = ExtractionOutput(
        research_question=research_question,
        method=method_text,
        datasets=datasets_list if datasets_list else None,
        metrics=metrics_list if metrics_list else None,
        key_finding=key_finding,
    )

    return ExtractionExample(
        chunk_id=f"{paper_id}_{section_label}",
        arxiv_id=paper_id,
        section_label=section_label,
        input_text=user_content,  # Store full prompt for reference
        output=output,
        system_prompt=system_prompt,
    )


def convert_qasper(
    limit: int = 2000,
    output_dir: Path = Path("./data/extraction_dataset"),
    val_split: float = 0.1,
    dry_run: bool = False,
) -> dict:
    """Convert QASPER dataset to extraction format."""
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
    skipped_short_chunk = 0

    for i, paper in enumerate(dataset):
        if i % 100 == 0 and i > 0:
            print(f"Processing paper {i+1}/{len(dataset)}...")

        full_text = paper.get("full_text")
        if not full_text:
            skipped_no_full_text += 1
            continue

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

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
                skipped_short_chunk += 1
                continue

            if section_name == "method" and len(section_text.strip()) < 200:
                skipped_short_chunk += 1
                continue

            if section_name in ("results", "evaluation"):
                if not re.search(r"\d", section_text):
                    skipped_short_chunk += 1
                    continue

            example = generate_extraction_example(
                paper=paper,
                chunk_text=section_text,
                section_label=section_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                paper_idx=i,
            )

            if example:
                jsonl_line = json.dumps(example.to_sft_row())
                try:
                    parsed = json.loads(jsonl_line)
                    if parsed.get("messages"):
                        examples.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Generated invalid JSONL for {example.chunk_id}")

    # Deduplicate by chunk_id (now unique due to paper_idx fallback)
    seen_chunk_ids = set()
    unique_examples = []
    for ex in examples:
        if ex.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(ex.chunk_id)
            unique_examples.append(ex)
        else:
            print(f"Warning: Duplicate chunk_id {ex.chunk_id}, skipping")
    examples = unique_examples

    print("\nConversion complete:")
    print(f"  Total examples generated: {len(examples)}")
    print(f"  Skipped (no full_text): {skipped_no_full_text}")
    print(f"  Skipped (chunk too short or no numbers): {skipped_short_chunk}")

    if len(examples) < 30:
        print("ERROR: Generated too few examples. Check the QASPER schema and extraction logic.")
        return {
            "total_examples": 0,
            "train_size": 0,
            "val_size": 0,
            "dry_run": dry_run,
        }

    paper_to_examples = defaultdict(list)
    for ex in examples:
        paper_to_examples[ex.arxiv_id].append(ex)

    paper_ids = list(paper_to_examples.keys())
    random.shuffle(paper_ids)

    val_paper_count = max(1, int(len(paper_ids) * val_split))
    val_paper_ids = set(paper_ids[:val_paper_count])

    val_examples = []
    train_examples = []
    for pid, exs in paper_to_examples.items():
        if pid in val_paper_ids:
            val_examples.extend(exs)
        else:
            train_examples.extend(exs)

    print(f"  Train papers: {len(paper_ids) - val_paper_count} ({len(train_examples)} examples)")
    print(f"  Val papers: {val_paper_count} ({len(val_examples)} examples)")

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
