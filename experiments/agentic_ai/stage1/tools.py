from calculator import calculator
from weather import get_weather

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to calculate."
                }
            },
            "required": ["expression"]
        }
    }
}

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for."
                }
            },
            "required": ["city"]
        }
    }
}

tools = [
    calculator_tool,
    weather_tool,
]

tool_registry = {
    "calculator": calculator,
    "get_weather": get_weather,
}


def execute_tool(tool_name, arguments):
    if tool_name not in tool_registry:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = tool_registry[tool_name]

    return tool(**arguments)
