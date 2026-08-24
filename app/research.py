"""
app/research.py

AI concerns only: question in, structured answer out. This module
knows nothing about HTTP, request/response objects, or the command
line — it's a plain function that talks to the LLM.

    question
       |
       v
    OpenRouter
       |
       v
    structured result (dict)

Both app/main.py (the FastAPI HTTP layer) and the top-level cli.py
import ask_research_question() from here, so the LLM logic exists in
exactly one place regardless of which interface is calling it:

    HTTP concerns          AI concerns
         |                     |
         v                     v
      main.py              research.py
         |                     |
         └──────────┬──────────┘
                     v
                  result

Setup:
    pip install openai python-dotenv
    export OPENROUTER_API_KEY=sk-or-...
    (or put OPENROUTER_API_KEY=sk-or-... in a .env file at the project root)
"""

import os
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
    response, with a message intended to be shown directly to the caller
    (CLI or API layer) — this function itself doesn't print or know
    about HTTP status codes.
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