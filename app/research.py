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

import json
import logging
import os
import time
import uuid

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.metrics import (
    record_request_duration,
    record_request_failed,
    record_request_started,
    record_request_success,
    record_retry,
    record_usage,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
                    "description": (
                        "A clear, well-reasoned answer to the research question, "
                        "a few paragraphs at most."
                    ),
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "3-6 short bullet-point takeaways that summarize "
                        "or support the answer."
                    ),
                },
            },
            "required": ["answer", "key_points"],
        },
    },
}


# --- Retry policy ---------------------------------------------------
#
# Retry temporary failures, but do not retry permanent request/auth errors.
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Research failed: no API key found. "
            "Set the OPENROUTER_API_KEY environment variable."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _call_llm_with_retries(
    client: OpenAI,
    question: str,
    request_id: str,
):
    """Call the chat completions endpoint, retrying temporary failures.

    Uses exponential backoff: 1s, 2s, 4s, ... between attempts.
    Only retries the exception types in RETRYABLE_EXCEPTIONS.
    """

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
                    "LLM usage: request_id=%s prompt_tokens=%s "
                    "completion_tokens=%s total_tokens=%s",
                    request_id,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )

                record_usage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
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
                record_retry()

                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))

                logger.warning(
                    "Attempt %d/%d failed after %.2fs (%s: %s); "
                    "retrying in %ds",
                    attempt,
                    MAX_ATTEMPTS,
                    duration,
                    type(e).__name__,
                    e,
                    delay,
                )

                time.sleep(delay)
            else:
                logger.error(
                    "Attempt %d/%d failed after %.2fs (%s: %s); "
                    "no attempts remaining",
                    attempt,
                    MAX_ATTEMPTS,
                    duration,
                    type(e).__name__,
                    e,
                )

        except APIStatusError as e:
            duration = time.perf_counter() - start_time

            logger.error(
                "Non-retryable status error after %.2fs (%s: %s); giving up",
                duration,
                type(e).__name__,
                e,
            )

            raise RuntimeError(
                f"Research failed: the AI service rejected the request ({e})."
            ) from e

        except Exception as e:
            duration = time.perf_counter() - start_time

            logger.error(
                "Unexpected error after %.2fs (%s: %s); giving up",
                duration,
                type(e).__name__,
                e,
            )

            raise RuntimeError(
                "Research failed: unable to contact the AI service."
            ) from e

    logger.error("All %d attempts exhausted; giving up", MAX_ATTEMPTS)

    raise RuntimeError(
        f"Research failed: unable to contact the AI service "
        f"after {MAX_ATTEMPTS} attempts."
    ) from last_error


def ask_research_question(question: str) -> dict:
    """Send the question to the LLM and return a structured result dict.

    Raises RuntimeError on any failure to contact or parse the model's
    response, with a message intended to be shown directly to the caller
    (CLI or API layer) — this function itself doesn't print or know
    about HTTP status codes.
    """

    request_id = str(uuid.uuid4())
    request_start = time.perf_counter()

    record_request_started()

    logger.info(
        "Research request started: request_id=%s",
        request_id,
    )

    try:
        client = get_client()
        response = _call_llm_with_retries(client, question, request_id)

        try:
            message = response.choices[0].message
            tool_call = message.tool_calls[0]
            data = json.loads(tool_call.function.arguments)

        except (
            IndexError,
            AttributeError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            logger.error(
                "Research response parsing failed: request_id=%s error_type=%s",
                request_id,
                type(e).__name__,
            )

            raise RuntimeError(
                "Research failed: the AI service returned an unexpected response."
            ) from e

        record_request_success()

        return {
            "question": question,
            "answer": data.get("answer", ""),
            "key_points": data.get("key_points", []),
        }

    except RuntimeError:
        record_request_failed()
        raise

    finally:
        duration = time.perf_counter() - request_start
        record_request_duration(duration)
