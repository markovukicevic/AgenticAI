#!/usr/bin/env python3
"""
research.py

Accepts a research question on the command line, sends it to an LLM via
OpenRouter, and prints back a structured JSON research answer.

Usage:
    python research.py "Why is nuclear energy difficult to scale?"

Setup:
    pip install openai python-dotenv
    export OPENROUTER_API_KEY=sk-or-...
    (or put OPENROUTER_API_KEY=sk-or-... in a .env file next to this script)
"""

import os
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError

load_dotenv()  # pulls OPENROUTER_API_KEY from a .env file, if present

# Any model OpenRouter serves that supports tool calling works here.
# See https://openrouter.ai/models for the full list.
MODEL = "openrouter/free"

SYSTEM_PROMPT = (
    "You are a careful research assistant. Given a research question, "
    "you provide a clear, accurate, well-reasoned answer along with a "
    "short list of key points that support or summarize that answer. "
    "Be concise but substantive. If a question is ambiguous or has no "
    "settled answer, say so honestly rather than inventing certainty."
)

# We force the model to reply through a single function call so the
# output is guaranteed to be valid, structured JSON rather than
# free-form text we'd have to hope was parseable.
ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_research_answer",
        "description": "Submit the structured research answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A clear, well-reasoned answer to the research question, a few paragraphs at most.",
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-6 short bullet-point takeaways that summarize or support the answer.",
                },
            },
            "required": ["answer", "key_points"],
        },
    },
}


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Research failed: no API key found. "
            "Set the OPENROUTER_API_KEY environment variable."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def ask_research_question(question: str) -> dict:
    """Send the question to the LLM and return a structured result dict.

    Raises RuntimeError on any failure to contact or parse the model's
    response, with a message intended to be shown directly to the user.
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=[ANSWER_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_research_answer"}},
        )
    except (APIError, APIConnectionError):
        raise RuntimeError("Research failed: unable to contact the AI service.")
    except Exception:
        raise RuntimeError("Research failed: unable to contact the AI service.")

    try:
        # openai SDK response objects are pydantic models; dump everything.
        print(response.model_dump_json(indent=2), file=sys.stderr)
    except Exception:
        # Fallback in case model_dump_json isn't available for some reason.
        print(response, file=sys.stderr)
        print(response.choices[0].message.tool_calls, file=sys.stderr)

    try:
        message = response.choices[0].message
        tool_call = message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
    except (IndexError, AttributeError, TypeError, json.JSONDecodeError):
        raise RuntimeError("Research failed: the AI service returned an unexpected response.")

    return {
        "question": question,
        "answer": data.get("answer", ""),
        "key_points": data.get("key_points", []),
    }


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