#!/usr/bin/env python3
"""Offline Ragas baseline. Does not enforce ragas_faithfulness_threshold."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from ragas.embeddings.base import BaseRagasEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.evaluation.gold import load_gold_dataset
from src.paperlens.evaluation.ragas_eval import (
    empty_context_scores,
    generation_rows,
    sample_fields,
)
from src.paperlens.evaluation.ragas_gate import (
    evaluate_faithfulness_gate,
    normalize_score,
    summarize_metric,
)

EXPECTED_CI_GENERATION_ROWS = 75

logger = logging.getLogger("paperlens.eval")


class BgeQueryEmbeddings(BaseRagasEmbedding):
    """Ragas embeddings over the already-loaded EmbeddingModel.

    Answer relevancy embeds the user question and questions generated from
    the answer. embed_query applies the BGE query prefix. Do not call embed_chunks.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._model.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas evaluation for generation gold")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Enforce the faithfulness threshold and require complete score coverage.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--save-answers", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("eval/ragas_baseline.checkpoint.jsonl"),
        help="Per-row score checkpoint used with --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a matching checkpoint and reuse saved answers.",
    )
    parser.add_argument("--top-k", type=_positive_int, default=None)
    parser.add_argument("--allow-langfuse", action="store_true")

    args = parser.parse_args(argv)

    if args.fail_under is not None and normalize_score(args.fail_under) is None:
        parser.error("--fail-under must be a finite number in [0, 1]")

    if args.output is None:
        args.output = Path(
            "eval/ragas_ci_result.json"
            if args.fail_under is not None
            else "eval/ragas_baseline.json"
        )

    return args


def _standardize_scored_rows(scored: list[dict], names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for record in scored:
        safe_record = {
            "id": record.get("id"),
            "n_contexts": record.get("n_contexts"),
            "refused": record.get("refused"),
        }
        for name in names:
            safe_record[name] = normalize_score(record.get(name))
        rows.append(safe_record)
    return rows


def _summarize_results(
    scored: list[dict],
    names: list[str],
    *,
    expected_rows: int,
    threshold: float | None,
) -> tuple[list[dict], dict[str, dict], dict | None, int]:
    safe_rows = _standardize_scored_rows(scored, names)
    summaries = {name: summarize_metric(safe_rows, name, expected_rows) for name in names}

    if threshold is None:
        return safe_rows, summaries, None, 0

    faithfulness = summaries.get("faithfulness")
    if faithfulness is None:
        raise RuntimeError("Ragas did not produce a faithfulness metric")

    gate = evaluate_faithfulness_gate(
        faithfulness,
        threshold=threshold,
        expected_rows=expected_rows,
    )
    return safe_rows, summaries, gate, 0 if gate["gate_passed"] else 1


def _load_or_create_checkpoint(path: Path, metadata: dict, *, resume: bool) -> dict[str, dict]:
    if not path.exists():
        if resume:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"_metadata": metadata}) + "\n")
        return {}
    if not resume:
        raise FileExistsError(
            f"Checkpoint already exists; pass --resume or choose another path: {path}"
        )

    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines or json.loads(lines[0]).get("_metadata") != metadata:
        raise ValueError(f"Checkpoint metadata does not match this evaluation: {path}")

    records: dict[str, dict] = {}
    for line in lines[1:]:
        record = json.loads(line)
        records[record["id"]] = record
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _has_complete_scores(record: dict | None, names: list[str]) -> bool:
    return record is not None and all(
        normalize_score(record.get(name)) is not None for name in names
    )


def _dry_run(args: argparse.Namespace) -> int:
    from src.paperlens.settings import get_settings

    path = args.gold or Path(get_settings().gold_dataset_path)
    if not path.exists():
        print(f"Gold file not found: {path}", file=sys.stderr)
        return 1
    rows = generation_rows(load_gold_dataset(path))
    limit = 2 if args.limit is None else args.limit
    for row in rows[:limit]:
        print(
            f"{row.get('id')} ground_truth={bool(str(row.get('ground_truth') or '').strip())} "
            f"query={row.get('query', '')[:80]}"
        )
    print(f"dry-run rows={min(limit, len(rows))} generation_rows={len(rows)}")
    return 0


def _judge_config(settings):
    model = settings.ragas_judge_model or settings.llm_model or settings.ollama_model
    base_url = settings.ragas_judge_base_url or settings.llm_api_base_url
    api_key = settings.ragas_judge_api_key or settings.llm_api_key
    return model, base_url, api_key


def _build_metrics(judge_model: str, base_url: str, api_key: str, embedder):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=5)
    llm = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=4096,
        reasoning_effort="low",
    )
    embeddings = BgeQueryEmbeddings(embedder)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecision(llm=llm)
    precision_params = set(inspect.signature(precision.ascore).parameters)
    if "reference" not in precision_params:
        from ragas.metrics.collections import ContextPrecisionWithReference

        precision = ContextPrecisionWithReference(llm=llm)
    return [faithfulness, relevancy, precision]


async def _score_metric(metric, fields: dict, semaphore: asyncio.Semaphore):
    params = inspect.signature(metric.ascore).parameters
    kwargs = {name: fields[name] for name in params if name in fields}
    delay = 2.0

    async with semaphore:
        for attempt in range(3):
            try:
                result = await metric.ascore(**kwargs)
                if result.value is None:
                    return None
                return float(result.value)
            except Exception as exc:
                logger.warning(
                    "%s attempt %d failed (%s)",
                    metric.name,
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt == 2:
                    raise RuntimeError(f"{metric.name} scoring failed after 3 attempts") from exc
                await asyncio.sleep(delay)
                delay *= 2

    raise RuntimeError(f"{metric.name} scoring failed")


def _load_answers(path: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for row in load_gold_dataset(path):
        found[row["id"]] = row
    return found


async def _run(args: argparse.Namespace) -> int:
    from src.paperlens.api.prompt_loader import PromptLoader
    from src.paperlens.api.rag import RAGService, get_cached_embedding_model
    from src.paperlens.api.schemas import QueryRequest
    from src.paperlens.llm import get_llm_provider
    from src.paperlens.settings import get_settings

    settings = get_settings()
    if not args.answers and settings.llm_provider == "ollama":
        print(
            "Refusing to generate with Ollama while the embedding model is loaded. "
            "Set LLM_PROVIDER=openai_compat.",
            file=sys.stderr,
        )
        return 2
    judge_model, base_url, api_key = _judge_config(settings)
    if not base_url or not api_key:
        print(
            "Refusing to load a local judge. Set RAGAS_JUDGE_MODEL and either "
            "RAGAS_JUDGE_BASE_URL / RAGAS_JUDGE_API_KEY or "
            "LLM_API_BASE_URL / LLM_API_KEY.",
            file=sys.stderr,
        )
        return 2
    # if "gpt-oss" in judge_model:
    #     print(
    #         "Refusing gpt-oss as the Ragas judge. llm_factory does not send "
    #         "reasoning_effort. Set RAGAS_JUDGE_MODEL=qwen/qwen3-8b.",
    #         file=sys.stderr,
    #     )
    #     return 2

    gold_path = args.gold or Path(settings.gold_dataset_path)
    all_rows = generation_rows(load_gold_dataset(gold_path))

    if not all_rows:
        print("No generation rows", file=sys.stderr)
        return 2

    if args.fail_under is not None:
        if len(all_rows) != EXPECTED_CI_GENERATION_ROWS:
            print(
                f"Gate requires exactly {EXPECTED_CI_GENERATION_ROWS} generation rows; "
                f"found {len(all_rows)}",
                file=sys.stderr,
            )
            return 2
        if args.limit is not None and args.limit < len(all_rows):
            print(
                "--limit cannot be used to run a partial fail-under evaluation",
                file=sys.stderr,
            )
            return 2

    rows = all_rows if args.limit is None else all_rows[: args.limit]
    if not rows:
        print("No rows selected for evaluation", file=sys.stderr)
        return 2

    metadata = {
        "dataset_version": "4.3.0",
        "gold_path": str(gold_path),
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "row_ids": [row["id"] for row in all_rows],
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model or settings.ollama_model,
        "llm_fallback_model": (
            settings.llm_fallback_model or settings.llm_model or settings.ollama_model
        ),
        "llm_base_url": settings.llm_api_base_url,
        "judge_model": judge_model,
        "judge_base_url": base_url,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": args.top_k or settings.retrieval_top_k,
    }
    try:
        scored_by_id = _load_or_create_checkpoint(args.checkpoint, metadata, resume=args.resume)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    embedder = get_cached_embedding_model(settings)
    metrics = _build_metrics(judge_model, base_url, api_key, embedder)
    names = [metric.name for metric in metrics]
    refusal = PromptLoader(settings).load_prompt().refusal_message
    answers = _load_answers(args.answers) if args.answers else {}
    rag = None
    if not args.answers or args.save_answers:
        rag = RAGService(settings, llm_provider=get_llm_provider(settings))

    semaphore = asyncio.Semaphore(1)
    log_path = Path("eval/ragas_baseline.jsonl")

    for row in rows:
        prior = scored_by_id.get(row["id"])
        if _has_complete_scores(prior, names):
            continue

        saved = answers.get(row["id"])
        generated = saved is None
        if saved is not None:
            answer = saved.get("response") or ""
            contexts = list(saved.get("retrieved_contexts") or [])
            chunk_ids = list(saved.get("retrieved_chunk_ids") or [])
        else:
            if args.answers and not args.save_answers:
                print(f"Missing saved answer for {row['id']}", file=sys.stderr)
                return 1
            if rag is None:
                rag = RAGService(settings, llm_provider=get_llm_provider(settings))
            response, contexts = await rag.query_with_contexts(
                QueryRequest(query=row["query"], top_k=args.top_k)
            )
            answer = response.answer or ""
            chunk_ids = [item.chunk_id for item in (response.citations or [])]

        refused = answer.strip() == refusal
        answer_record = {
            "id": row["id"],
            "response": answer,
            "retrieved_contexts": contexts,
            "retrieved_chunk_ids": chunk_ids,
        }
        if generated and args.save_answers:
            _append_jsonl(args.save_answers, answer_record)
            answers[row["id"]] = answer_record

        record = {
            "id": row["id"],
            "n_contexts": len(contexts),
            "refused": refused,
        }
        if not contexts:
            record.update(empty_context_scores(names))
        else:
            fields = sample_fields(row, answer, contexts)
            values = await asyncio.gather(
                *[_score_metric(metric, fields, semaphore) for metric in metrics]
            )
            for name, value in zip(names, values, strict=True):
                record[name] = normalize_score(value)
        scored_by_id[row["id"]] = record
        _append_jsonl(args.checkpoint, record)
        _append_jsonl(log_path, record)
        print(f"{row['id']} contexts={len(contexts)} refused={refused}")

    scored = [scored_by_id[row["id"]] for row in rows]
    # skipped_empty = sum(1 for record in scored if record["n_contexts"] == 0)

    expected_rows = len(rows)
    safe_rows, metric_summaries, gate, exit_code = _summarize_results(
        scored,
        names,
        expected_rows=expected_rows,
        threshold=args.fail_under,
    )
    faithfulness = metric_summaries["faithfulness"]

    import ragas

    payload = {
        "dataset_version": "4.3.0",
        "n_gold_generation": len(all_rows),
        "n_scored": len(scored),
        "n_valid": faithfulness["n_valid"],
        "n_invalid": faithfulness["n_invalid"],
        "invalid_ids": faithfulness["invalid_ids"],
        "threshold": args.fail_under,
        "threshold_enforced": args.fail_under is not None,
        "coverage_ratio": faithfulness["coverage_ratio"],
        "gate_passed": gate["gate_passed"] if gate is not None else None,
        "gate": gate,
        "limit": args.limit,
        "retrieval_top_k": args.top_k or settings.retrieval_top_k,
        "ragas_version": ragas.__version__,
        "langfuse_enabled": False,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "metrics": metric_summaries,
        "rows": safe_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(json.dumps(payload["metrics"], indent=2, allow_nan=False))

    if gate is not None and not gate["gate_passed"]:
        print(
            f"Faithfulness gate failed: reason={gate['reason']} "
            f"mean={gate['mean']} threshold={gate['threshold']} "
            f"valid={gate['n_valid']}/{gate['n_expected']}",
            file=sys.stderr,
        )

    return exit_code


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    try:
        from src.paperlens.settings import get_settings

        settings = get_settings()
        if settings.langfuse_enabled and not args.allow_langfuse:
            print(
                "Refusing to score with Langfuse enabled. "
                "Run via make eval-ragas or pass --allow-langfuse.",
                file=sys.stderr,
            )
            return 2

        return asyncio.run(_run(args))
    except Exception as exc:
        logger.error("Evaluation failed (%s)", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
