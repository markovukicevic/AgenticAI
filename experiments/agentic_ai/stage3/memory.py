import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from schemas import MemoryExtraction


load_dotenv()


MODEL = "openrouter/free"


MEMORY_SYSTEM_PROMPT = """
You are a memory extraction component.

Your job is to extract useful, stable facts about the user
from the user's latest message.

Only extract facts that may be useful later in the conversation.

Examples of useful facts:
- the user's name
- preferences
- favorite things
- goals
- chosen options
- important ongoing context

Do not extract:
- ordinary questions
- temporary calculations
- greetings
- facts about the outside world
- assistant responses
- information that is not about the user

Return only valid JSON in this format:

{
    "facts": {
        "key": "value"
    }
}

If there is nothing worth remembering, return:

{
    "facts": {}
}

Use short snake_case keys.
"""


def get_memory_client() -> OpenAI:
    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY found."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def extract_memory(
    user_message: str,
) -> MemoryExtraction:
    client = get_memory_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": MEMORY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    content = (
        response.choices[0].message.content
        or '{"facts": {}}'
    )

    try:
        parsed = json.loads(
            content
        )

    except json.JSONDecodeError:
        return MemoryExtraction()

    try:
        return MemoryExtraction.model_validate(
            parsed
        )

    except Exception:
        return MemoryExtraction()