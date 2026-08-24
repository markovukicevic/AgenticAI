#!/usr/bin/env python3
"""
cli.py

CLI entry point for the research agent — the second interface on top
of app/research.py, alongside the HTTP API in app/main.py. Neither
interface contains any OpenRouter/prompt logic itself; they both just
call ask_research_question().

    CLI (this file) ──┐
                       ├──> app/research.py ──> OpenRouter
    HTTP (app/main.py)┘

Usage (run from the project root):
    python cli.py "Why is nuclear energy difficult to scale?"
    python cli.py --demo   (runs 5 built-in test questions)

Setup:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-...
    (or put OPENROUTER_API_KEY=sk-or-... in a .env file at the project root)
"""

import sys
import json

from app.research import ask_research_question

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
        print('Usage: python cli.py "your research question"')
        print('       python cli.py --demo   (runs 5 built-in test questions)')
        sys.exit(1)

    if sys.argv[1] == "--demo":
        run_demo()
        return

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python cli.py "your research question"')
        sys.exit(1)

    try:
        result = ask_research_question(question)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
