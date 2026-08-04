#!/usr/bin/env python3
"""Smoke test for prompt loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paperlens.api.prompt_loader import PromptLoader
from src.paperlens.settings import get_settings


def main() -> int:
    settings = get_settings()
    loader = PromptLoader(settings)
    prompt = loader.load_prompt()

    print(f"Prompt version: {prompt.version}")
    print(f"System prompt length: {len(prompt.system_prompt)} chars")
    print(f"User template length: {len(prompt.user_template)} chars")
    print(f"Refusal message: {prompt.refusal_message[:40]}...")

    # Test prompt building
    test_prompt = prompt.build_prompt(context="[chunk_001] Test context.", query="test question")
    print(f"Built prompt length: {len(test_prompt)} chars")

    print("Prompt loader OK")
    return 0


if __name__ == "__main__":
    exit(main())
