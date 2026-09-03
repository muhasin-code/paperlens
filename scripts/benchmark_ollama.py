#!/usr/bin/env python3
"""Benchmark Ollama models: tokens/sec, TTFT, peak RAM.

Usage:
    python scripts/benchmark_ollama.py --model phi4-mini --runs 3
    python scripts/benchmark_ollama.py --model llama3.2:1b --runs 3
    make benchmark                  # runs both models, 3 runs each
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import ollama
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paperlens.settings import get_settings

settings = get_settings()


@dataclass
class BenchmarkResult:
    model: str
    run: int
    prompt: str
    total_tokens: int
    ttft_ms: float
    tokens_per_sec: float
    peak_rss_mb: float
    total_latency_ms: float
    generated_text: str


PROMPTS = [
    "What is LoRA and how does it work?",
    "Explain the transformer architecture in simple terms.",
    "How do recent papers approach learning rate scheduling?",
    "What are the key differences between RNNs and Transformers?",
    "Summarize the attention mechanism in one paragraph.",
]


async def benchmark_model(model: str, runs: int, prompt: str) -> list[BenchmarkResult]:
    """Run benchmark for a single model/prompt combination."""
    client = ollama.AsyncClient(host=settings.ollama_base_url)

    results: list[BenchmarkResult] = []
    process = psutil.Process(os.getpid())

    for run in range(1, runs + 1):
        # Warm-up (not measured)
        await client.generate(model=model, prompt=prompt, stream=False, options={"num_predict": 1})

        # Measured run
        start_wall = time.perf_counter()
        start_rss = process.memory_info().rss

        ttft_recorded = False
        ttft_ms = 0.0
        token_count = 0
        generated_chunks: list[str] = []

        async for chunk in await client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={"temperature": 0.1, "num_predict": 256},
        ):
            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - start_wall) * 1000
                ttft_recorded = True

            token = chunk.get("response", "")
            if token:
                generated_chunks.append(token)
                token_count += len(token) // 4  # rough token estimate (chars/4)

        total_latency_ms = (time.perf_counter() - start_wall) * 1000
        peak_rss_mb = start_rss / (1024 * 1024)
        tokens_per_sec = (token_count / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0.0

        results.append(
            BenchmarkResult(
                model=model,
                run=run,
                prompt=prompt,
                total_tokens=token_count,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
                peak_rss_mb=peak_rss_mb,
                total_latency_ms=total_latency_ms,
                generated_text="".join(generated_chunks),
            )
        )

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama models (streaming).")
    parser.add_argument(
        "--model",
        action="append",
        help="Model name (repeat for multiple). Default: both phi4-mini and llama3.2:1b",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per model per prompt (default: 3)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks.json"), help="Output JSON path"
    )
    args = parser.parse_args()

    models = args.model or [settings.ollama_model]
    all_results: list[BenchmarkResult] = []

    print(f"Benchmarking models: {models}")
    print(f"Runs per prompt: {args.runs}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    for model in models:
        print(f"\n Model: {model}")
        for prompt in PROMPTS:
            print(f"  Prompt: {prompt[:50]}...")
            try:
                results = await benchmark_model(model, args.runs, prompt)
                all_results.extend(results)
                for r in results:
                    print(
                        f"    Run {r.run}: {r.tokens_per_sec:.1f} tok/s, "
                        f"TTFT {r.ttft_ms:.1f} ms, Peak RSS {r.peak_rss_mb:.1f} MB"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"    Failed: {exc}")

    # Save raw results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
    print(f"\n Raw results saved to {args.output}")

    # Summary table
    print("\n=== Summary (median across runs & prompts) ===")
    for model in models:
        model_results = [r for r in all_results if r.model == model]
        if not model_results:
            continue
        tok_s = statistics.median(r.tokens_per_sec for r in model_results)
        ttft = statistics.median(r.ttft_ms for r in model_results)
        rss = statistics.median(r.peak_rss_mb for r in model_results)
        lat = statistics.median(r.total_latency_ms for r in model_results)
        print(
            f"{model:15s} | {tok_s:6.1f} tok/s | TTFT {ttft:6.1f} ms | RSS {rss:5.1f} MB | Latency {lat:7.1f} ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
