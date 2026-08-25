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
import time
import logging
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    APIStatusError,
)

load_dotenv()  # pulls OPENROUTER_API_KEY from a .env file, if present


# Any model OpenRouter serves that supports tool calling works here.
# See https://openrouter.ai/models for the full list.
MODEL = "nvidia/nemotron-3.5-lightning:free"

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

# --- Retry policy ---------------------------------------------------
#
# Not every failure is worth retrying. A 429 (rate limited) or a 5xx
# (upstream having a bad moment) might well succeed on the next try.
# A 400 (we sent a malformed request) or 401 (bad API key) will fail
# exactly the same way every time — retrying just burns time and, on
# a paid API, money, for no benefit.
#
# The openai SDK already classifies HTTP error responses into distinct
# exception types by status code, so we don't need to inspect status
# codes ourselves — we just decide which *exception types* are worth
# retrying:
#
#   RateLimitError      -> HTTP 429, temporary, retry
#   InternalServerError -> HTTP 5xx (502/503/...), temporary, retry
#   APIConnectionError  -> network/connection failure, retry
#   APITimeoutError     -> subclass of APIConnectionError, retry
#
# Everything else that's an APIStatusError (400, 401, 403, 404, 409,
# 422, ...) reflects something wrong with the request itself or our
# credentials, and will not become valid by trying again -> no retry.
RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1  # attempt 1 fails -> wait 1s, attempt 2 fails -> wait 2s, ...


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Nothing a retry could fix here — the key is either set or it
        # isn't. Fail immediately, before we even build a client.
        raise RuntimeError(
            "Research failed: no API key found. "
            "Set the OPENROUTER_API_KEY environment variable."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def _call_llm_with_retries(client: OpenAI, question: str, request_id: str):
    """Call the chat completions endpoint, retrying temporary failures.

    Uses exponential backoff: 1s, 2s, 4s, ... between attempts (not
    after the final attempt, since there's nothing left to wait for).
    Only retries the specific exception types in RETRYABLE_EXCEPTIONS;
    anything else (bad request, bad auth, ...) is raised immediately
    on the first failure, since trying again wouldn't change anything.
    """
    logger.info("Starting research request: request_id=%s",
                request_id)

    logger.info(
        "Sending research request request_id=%s to model=%s",
        request_id,
        MODEL,
    )

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start_time = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tools=[ANSWER_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_research_answer"},
                },
            )

            usage = response.usage

            if usage:
                logger.info(
                    "LLM usage: request_id=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    request_id,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )

            logger.info(
                "LLM response received: request_id=%s model=%s finish_reason=%s",
                request_id,
                MODEL,
                response.choices[0].finish_reason,
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "Research request succeeded: request_id=%s duration=%.2fs",
                request_id,
                duration,
            )

            return response

        except RETRYABLE_EXCEPTIONS as e:
            duration = time.perf_counter() - start_time
            last_error = e

            if attempt < MAX_ATTEMPTS:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))  # 1, 2, 4, ...
                logger.warning(
                    "Attempt %d/%d failed after %.2fs (%s: %s); retrying in %ds",
                    attempt, MAX_ATTEMPTS, duration, type(e).__name__, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Attempt %d/%d failed after %.2fs (%s: %s); no attempts remaining",
                    attempt, MAX_ATTEMPTS, duration, type(e).__name__, e,
                )
            # else: this was the last attempt, fall through to the loop
            # ending and raise below.

        except APIStatusError as e:
            duration = time.perf_counter() - start_time
            # A 4xx that isn't in RETRYABLE_EXCEPTIONS: bad request,
            # bad auth, etc. Retrying won't help, so give up right away
            # instead of burning two more attempts and several seconds.
            logger.error(
                "Non-retryable status error after %.2fs (%s: %s); giving up",
                duration, type(e).__name__, e,
            )
            raise RuntimeError(f"Research failed: the AI service rejected the request ({e}).")

        except Exception as e:
            duration = time.perf_counter() - start_time
            # Something unexpected (not an openai SDK error at all).
            # Treat as non-retryable — we don't know what it is, so we
            # don't know that trying again is safe or useful.
            logger.error(
                "Unexpected error after %.2fs (%s: %s); giving up",
                duration, type(e).__name__, e,
            )
            raise RuntimeError("Research failed: unable to contact the AI service.")

    # Exhausted every attempt on retryable errors — give up rather than
    # retry indefinitely and leave the caller waiting forever.
    logger.error("All %d attempts exhausted; giving up", MAX_ATTEMPTS)
    raise RuntimeError(
        f"Research failed: unable to contact the AI service after {MAX_ATTEMPTS} attempts."
    ) from last_error


def ask_research_question(question: str) -> dict:
    request_id = str(uuid.uuid4())
    """Send the question to the LLM and return a structured result dict.

    Raises RuntimeError on any failure to contact or parse the model's
    response, with a message intended to be shown directly to the caller
    (CLI or API layer) — this function itself doesn't print or know
    about HTTP status codes.
    """


    logger.info(
        "Research request started: request_id=%s",
        request_id,
    )

    client = get_client()
    response = _call_llm_with_retries(client, question, request_id)

    try:
        message = response.choices[0].message
        tool_call = message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
    except (IndexError, AttributeError, TypeError, json.JSONDecodeError):
        # The model responded, but not in the shape we asked for. This
        # is a different failure class from a network/upstream error:
        # asking again with the same input might just produce the same
        # bad response again, and it isn't "temporary" in the retry
        # sense. We surface it immediately rather than auto-retrying.
        raise RuntimeError("Research failed: the AI service returned an unexpected response.")

    return {
        "question": question,
        "answer": data.get("answer", ""),
        "key_points": data.get("key_points", []),
    }