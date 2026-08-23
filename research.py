#!/usr/bin/env python3
"""
research.py

CLI entry point. Accepts a research question on the command line and
prints back a structured JSON research answer.

All the actual LLM logic lives in research_service.py — this file is
just a thin wrapper: parse argv, call the service, print the result
(or a clean error message).

Usage:
    python research.py "Why is nuclear energy difficult to scale?"
    python research.py --demo   (runs 5 built-in test questions)

Setup:
    pip install openai python-dotenv
    export OPENROUTER_API_KEY=sk-or-...
    (or put OPENROUTER_API_KEY=sk-or-... in a .env file next to this script)
"""

import sys
import json

from research_service import ask_research_question

# A handful of test questions covering different kinds of difficulty,
# used by `--demo` to smoke-test the pipeline end to end.
DEMO_QUESTIONS = [
    ("Straightforward factual", "What is the boiling point of water at sea level?"),
    ("Ambiguous", "How long does it take to learn a language?"),
    ("Reasonable people disagree", "Should social media platforms be regulated like utilities?"),
    ("Complex technical", "How does gradient descent avoid getting stuck in local minima in deep neural networks?"),
    ("Likely to cause struggle", "What will the exact global population be on January 1, 2100?"),
]


def run_demo():
    """Run a fixed set of test questions and print each result (or error)."""
    for label, question in DEMO_QUESTIONS:
        print("=" * 70)
        print(f"[{label}] {question}")
        print("=" * 70)
        try:
            result = ask_research_question(question)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except RuntimeError as e:
            print(str(e))
        print()


def main():
    if len(sys.argv) < 2:
        print('Usage: python research.py "your research question"')
        print('       python research.py --demo   (runs 5 built-in test questions)')
        sys.exit(1)

    if sys.argv[1] == "--demo":
        run_demo()
        return

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python research.py "your research question"')
        sys.exit(1)

    try:
        result = ask_research_question(question)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()