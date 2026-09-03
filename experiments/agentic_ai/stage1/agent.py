import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import execute_tool, tools


load_dotenv()


MODEL = "openrouter/free"


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY found. "
            "Make sure your .env file contains OPENROUTER_API_KEY."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def run_agent(user_question: str) -> str:
    client = get_client()

    messages = [
        {
            "role": "user",
            "content": user_question,
        }
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        # ---------------------------------------------------------
        # Case 1: The model has finished and does not need a tool.
        # ---------------------------------------------------------
        if not message.tool_calls:
            return message.content

        # ---------------------------------------------------------
        # Case 2: The model wants to use one or more tools.
        # ---------------------------------------------------------

        # Preserve the assistant's tool-call message in the
        # conversation history.
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = execute_tool(
                tool_name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )


if __name__ == "__main__":
    question = input("Ask the agent: ")

    answer = run_agent(question)

    print()
    print("Agent:", answer)