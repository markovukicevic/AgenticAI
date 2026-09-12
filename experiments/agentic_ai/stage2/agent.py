import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from memory import extract_memory
from memory_store import load_memory, save_memory
from schemas import AgentState
from tools import execute_tool, tools

load_dotenv()

MODEL = "openrouter/free"

MAX_ITERATIONS = 10
MAX_CONTEXT_TURNS = 5

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

        llm_messages = (
            state.build_llm_messages(
                max_recent_turns=(
                    MAX_CONTEXT_TURNS
                )
            )
        )

        print(
            "Context messages:",
            len(llm_messages),
        )

        print(
            "Stored conversation messages:",
            len(state.messages),
        )

        response = (
            client.chat.completions.create(
                model=MODEL,
                messages=llm_messages,
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
    state = AgentState(
        session_memory=load_memory()
    )

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
        ).strip()

        if not user_question:
            continue

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

        memory_extraction = extract_memory(
            user_question
        )

        print(
            "Extracted memory:",
            memory_extraction.facts,
        )

        for key, value in (
                memory_extraction.facts.items()
        ):
            state.remember(
                key,
                value,
            )

        if memory_extraction.facts:
            save_memory(
                state.session_memory
            )

        print(
            "Session memory:",
            state.session_memory,
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
