#!/usr/bin/env python3
"""Offline Ragas baseline. Does not enforce ragas_faithfulness_threshold."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.evaluation.gold import load_gold_dataset
from src.paperlens.evaluation.ragas_eval import (
    empty_context_scores,
    generation_rows,
    mean_scores,
    sample_fields,
)

logger = logging.getLogger("paperlens.eval")


class BgeQueryEmbeddings:
    """Ragas embeddings over the already-loaded EmbeddingModel.

    Answer relevancy embeds the user question and questions generated from
    the answer. embed_query applies BGE_QUERY_PREFIX. Do not call embed_chunks.
    Implements both the collections interface (embed_text) and the legacy one
    (aembed_query) so a minor Ragas bump does not load a second model.
    """

    def __init__(self, model) -> None:
        self._model = model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._model.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_queries(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas baseline for generation gold")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/ragas_baseline.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--save-answers", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--allow-langfuse", action="store_true")
    return parser.parse_args(argv)


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
    llm = llm_factory(judge_model, provider="openai", client=client, temperature=0)
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
                return float(result.value)
            except Exception as exc:
                logger.warning("%s attempt %d failed: %s", metric.name, attempt + 1, exc)
                if attempt == 2:
                    return None
                await asyncio.sleep(delay)
                delay *= 2
    return None


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
            "Refusing to load a local judge. Set RAGAS_JUDGE_MODEL and the Groq "
            "base URL / API key (LLM_API_BASE_URL, LLM_API_KEY).",
            file=sys.stderr,
        )
        return 2
    if "gpt-oss" in judge_model:
        print(
            "Refusing gpt-oss as the Ragas judge. llm_factory does not send "
            "reasoning_effort. Set RAGAS_JUDGE_MODEL=qwen/qwen3-8b.",
            file=sys.stderr,
        )
        return 2

    gold_path = args.gold or Path(settings.gold_dataset_path)
    rows = generation_rows(load_gold_dataset(gold_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No generation rows", file=sys.stderr)
        return 1

    embedder = get_cached_embedding_model(settings)
    metrics = _build_metrics(judge_model, base_url, api_key, embedder)
    names = [metric.name for metric in metrics]
    refusal = PromptLoader(settings).load_prompt().refusal_message
    answers = _load_answers(args.answers) if args.answers else None
    rag = None
    if answers is None:
        rag = RAGService(settings, llm_provider=get_llm_provider(settings))

    semaphore = asyncio.Semaphore(2)
    scored: list[dict] = []
    answer_lines: list[dict] = []
    skipped_empty = 0
    log_path = Path("eval/ragas_baseline.jsonl")

    for row in rows:
        if answers is None:
            response, contexts = await rag.query_with_contexts(
                QueryRequest(query=row["query"], top_k=args.top_k)
            )
            answer = response.answer or ""
            chunk_ids = [item.chunk_id for item in (response.citations or [])]
        else:
            saved = answers.get(row["id"])
            if saved is None:
                print(f"Missing saved answer for {row['id']}", file=sys.stderr)
                return 1
            answer = saved.get("response") or ""
            contexts = list(saved.get("retrieved_contexts") or [])
            chunk_ids = list(saved.get("retrieved_chunk_ids") or [])

        refused = answer.strip() == refusal
        record = {
            "id": row["id"],
            "n_contexts": len(contexts),
            "refused": refused,
        }
        if not contexts:
            skipped_empty += 1
            record.update(empty_context_scores(names))
        else:
            fields = sample_fields(row, answer, contexts)
            values = await asyncio.gather(
                *[_score_metric(metric, fields, semaphore) for metric in metrics]
            )
            for name, value in zip(names, values, strict=True):
                record[name] = value
        scored.append(record)
        answer_lines.append(
            {
                "id": row["id"],
                "response": answer,
                "retrieved_contexts": contexts,
                "retrieved_chunk_ids": chunk_ids,
            }
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"{row['id']} contexts={len(contexts)} refused={refused}")

    if args.save_answers:
        args.save_answers.parent.mkdir(parents=True, exist_ok=True)
        with args.save_answers.open("w", encoding="utf-8") as handle:
            for line in answer_lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    import ragas

    payload = {
        "dataset_version": "4.3.0",
        "gold_path": str(gold_path),
        "n_gold_generation": len(generation_rows(load_gold_dataset(gold_path))),
        "n_scored": len(scored),
        "n_skipped_empty_context": skipped_empty,
        "limit": args.limit,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model or settings.ollama_model,
        "judge_model": judge_model,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": args.top_k or settings.retrieval_top_k,
        "ragas_version": ragas.__version__,
        "langfuse_enabled": False,
        "threshold_enforced": False,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "metrics": mean_scores(scored, names),
        "rows": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
