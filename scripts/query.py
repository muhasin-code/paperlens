#!/usr/bin/env python3
"""CLI smoke test for the /query endpoint.

Usage:
    python scripts/query.py "How do recent papers approach learning rate scheduling?"
    python scripts/query.py --top-k 3 --model llama3.2:1b "What is LoRA?"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.paperlens.settings import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Query the PaperLens RAG API.")
    parser.add_argument("query", type=str, help="Natural language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument("--model", type=str, default=None, help="Override Ollama model.")
    parser.add_argument(
        "--host", type=str, default="localhost", help="API host (default: localhost)."
    )
    parser.add_argument("--port", type=int, default=None, help="API port (default from settings).")
    args = parser.parse_args()

    settings = get_settings()
    host = args.host
    port = args.port or settings.api_port
    url = f"http://{host}:{port}/query"

    payload = {"query": args.query}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.model is not None:
        payload["model"] = args.model

    print(f"→ POST {url}")
    print(f"  Query: {args.query}")
    if args.top_k:
        print(f"  top_k: {args.top_k}")
    if args.model:
        print(f"  model: {args.model}")
    print()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            print(f" HTTP {exc.response.status_code}: {exc.response.text}")
            sys.exit(1)
        except httpx.RequestError as exc:
            print(f" Request failed: {exc}")
            print("  Is the server running? Start with: make run")
            sys.exit(1)

    print(" Response received")
    print(f"  Answer: {data['answer'][:200]}{'...' if len(data['answer']) > 200 else ''}")
    print(f"  Confidence: {data['confidence']:.3f}")
    print(f"  Citations: {len(data['citations'])}")
    print(f"  Retrieval time: {data['retrieval_time_ms']:.1f} ms")
    print(f"  Generation time: {data['generation_time_ms']:.1f} ms")
    print(f"  Total time: {data['total_time_ms']:.1f} ms")
    print()
    print("Full JSON:")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
