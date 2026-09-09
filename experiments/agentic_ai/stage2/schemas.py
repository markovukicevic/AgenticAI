from typing import Any, Callable, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field


class CalculatorInput(BaseModel):
    expression: str


class CalculatorResult(BaseModel):
    expression: str
    result: float


class WeatherInput(BaseModel):
    city: str
    units: Literal[
        "celsius",
        "fahrenheit",
    ] = "celsius"


class WeatherResult(BaseModel):
    city: str
    temperature: float
    unit: Literal[
        "celsius",
        "fahrenheit",
    ]
    condition: str


class TimeInput(BaseModel):
    city: str


class TimeResult(BaseModel):
    city: str
    time: str
    timezone: str


class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class AgentState(BaseModel):
    messages: List[Any] = Field(
        default_factory=list
    )

    def add_user_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )


class ToolDefinition(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    name: str
    description: str
    function: Callable
    input_model: Type[BaseModel]
    output_model: Optional[
        Type[BaseModel]
    ] = None

    def to_llm_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": (
                    self.input_model.model_json_schema()
                ),
            },
        }

    def execute(
        self,
        arguments: dict,
    ) -> ToolResult:
        try:
            validated_arguments = (
                self.input_model(
                    **arguments
                )
            )

            result = self.function(
                **validated_arguments.model_dump()
            )

            if self.output_model is not None:
                if isinstance(
                    result,
                    self.output_model,
                ):
                    validated_result = result
                else:
                    validated_result = (
                        self.output_model.model_validate(
                            result
                        )
                    )

                result = (
                    validated_result.model_dump()
                )

            elif isinstance(
                result,
                BaseModel,
            ):
                result = result.model_dump()

            return ToolResult(
                success=True,
                data=result,
            )

        except Exception as error:
            return ToolResult(
                success=False,
                error=str(error),
            )