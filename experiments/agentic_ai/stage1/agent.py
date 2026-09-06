import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import execute_tool, tools


load_dotenv()


MODEL = "openrouter/free"

MAX_ITERATIONS = 10


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

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        print()
        print(f"--- LLM iteration {iteration + 1} ---")

        if message.tool_calls:
            print("Tool calls requested:")

            for tool_call in message.tool_calls:
                print(
                    f"  Tool: {tool_call.function.name}"
                )
                print(
                    f"  Arguments: {tool_call.function.arguments}"
                )

        else:
            print("No tool call. Final answer generated.")

            return message.content or (
                "The model returned no final answer."
            )

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as error:
                tool_result = (
                    f"Tool argument error: the arguments for "
                    f"'{tool_name}' were not valid JSON. "
                    f"Details: {error}"
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

                continue

            try:
                result = execute_tool(
                    tool_name,
                    arguments,
                )

            except ValueError as error:
                tool_result = (
                    f"Tool error: {error}"
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

                continue

            except Exception as error:
                tool_result = (
                    f"Tool '{tool_name}' failed during execution. "
                    f"Details: {error}"
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

                continue

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return (
        f"The agent stopped because it reached the maximum "
        f"iteration limit of {MAX_ITERATIONS}."
    )


if __name__ == "__main__":
    question = input("Ask the agent: ")

    answer = run_agent(question)

    print()
    print("Agent:", answer)