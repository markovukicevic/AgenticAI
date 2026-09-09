import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from schemas import AgentState
from tools import execute_tool, tools


load_dotenv()


MODEL = "openrouter/free"

MAX_ITERATIONS = 10


def get_client() -> OpenAI:
    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY found. "
            "Make sure your .env file contains "
            "OPENROUTER_API_KEY."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def run_agent(
    state: AgentState,
) -> str:
    client = get_client()

    for iteration in range(
        MAX_ITERATIONS
    ):
        response = (
            client.chat.completions.create(
                model=MODEL,
                messages=state.messages,
                tools=tools,
            )
        )

        message = (
            response.choices[0].message
        )

        print()
        print(
            f"--- LLM iteration "
            f"{iteration + 1} ---"
        )

        if not message.tool_calls:
            print(
                "No tool call. "
                "Final answer generated."
            )

            final_answer = (
                message.content
                or
                "The model returned no "
                "final answer."
            )

            state.add_assistant_message(
                final_answer
            )

            return final_answer

        print(
            "Tool calls requested:"
        )

        for tool_call in (
            message.tool_calls
        ):
            print(
                f"  Tool: "
                f"{tool_call.function.name}"
            )

            print(
                f"  Arguments: "
                f"{tool_call.function.arguments}"
            )

        state.messages.append(
            message
        )

        for tool_call in (
            message.tool_calls
        ):
            tool_name = (
                tool_call.function.name
            )

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except (
                json.JSONDecodeError
            ) as error:
                tool_result = {
                    "success": False,
                    "data": None,
                    "error": (
                        "Invalid JSON arguments: "
                        f"{error}"
                    ),
                }

                state.add_tool_result(
                    tool_call.id,
                    json.dumps(
                        tool_result
                    ),
                )

                continue

            result = execute_tool(
                tool_name,
                arguments,
            )

            if result.success:
                print(
                    "  Tool result: "
                    "success → "
                    f"{result.data}"
                )
            else:
                print(
                    "  Tool result: "
                    "failure → "
                    f"{result.error}"
                )

            state.add_tool_result(
                tool_call.id,
                json.dumps(
                    result.model_dump()
                ),
            )

    return (
        "The agent stopped because it "
        "reached the maximum iteration "
        f"limit of {MAX_ITERATIONS}."
    )


def run_conversation() -> None:
    state = AgentState()

    print(
        "Agent conversation started."
    )
    print(
        "Type 'exit' to stop."
    )

    while True:
        print()

        user_question = input(
            "You: "
        )

        if (
            user_question.lower()
            == "exit"
        ):
            print(
                "Conversation ended."
            )
            break

        state.add_user_message(
            user_question
        )

        answer = run_agent(
            state
        )

        print()
        print(
            "Agent:",
            answer,
        )


if __name__ == "__main__":
    run_conversation()