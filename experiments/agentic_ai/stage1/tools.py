from calculator import calculator
from schemas import CalculatorInput, TimeInput, WeatherInput
from clock import get_time
from weather import get_weather


def build_tool_schema(
    name: str,
    description: str,
    input_model: type,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_model.model_json_schema(),
        },
    }


tool_registry = {
    "calculator": {
        "function": calculator,
        "input_model": CalculatorInput,
        "description": "Calculate a mathematical expression.",
    },
    "get_weather": {
        "function": get_weather,
        "input_model": WeatherInput,
        "description": (
            "Get the current weather for a city. "
            "Use this tool when the user asks about weather."
        ),
    },
    "get_time": {
        "function": get_time,
        "input_model": TimeInput,
        "description": (
            "Get the current time for a city. "
            "Use this tool when the user asks about time."
        ),
    }
}


tools = [
    build_tool_schema(
        name=tool_name,
        description=tool["description"],
        input_model=tool["input_model"],
    )
    for tool_name, tool in tool_registry.items()
]


def execute_tool(tool_name, arguments):
    if tool_name not in tool_registry:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    tool = tool_registry[tool_name]

    validated_arguments = tool["input_model"](
        **arguments
    )

    return tool["function"](
        **validated_arguments.model_dump()
    )