from calculator import calculator
from clock import get_time
from schemas import (
    CalculatorInput,
    CalculatorResult,
    TimeInput,
    TimeResult,
    ToolDefinition,
    ToolResult,
    WeatherInput,
    WeatherResult,
)
from weather import get_weather


tool_registry = {
    "calculator": ToolDefinition(
        name="calculator",
        description="Calculate a mathematical expression.",
        function=calculator,
        input_model=CalculatorInput,
        output_model=CalculatorResult,
    ),
    "get_weather": ToolDefinition(
        name="get_weather",
        description=(
            "Get the current weather for a city. "
            "Use this tool when the user asks about weather."
        ),
        function=get_weather,
        input_model=WeatherInput,
        output_model=WeatherResult,
    ),
    "get_time": ToolDefinition(
        name="get_time",
        description=(
            "Get the current local time for a city. "
            "Use this tool when the user asks what time it is "
            "in a city."
        ),
        function=get_time,
        input_model=TimeInput,
        output_model=TimeResult,
    ),
}


tools = [
    tool.to_llm_schema()
    for tool in tool_registry.values()
]


def execute_tool(
    tool_name: str,
    arguments: dict,
) -> ToolResult:
    if tool_name not in tool_registry:
        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    return tool_registry[tool_name].execute(
        arguments
    )