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


class MemoryExtraction(BaseModel):
    facts: dict = Field(
        default_factory=dict
    )


class PlanStep(BaseModel):
    description: str
    completed: bool = False


class AgentPlan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(
        default_factory=list
    )


class AgentState(BaseModel):
    messages: List[Any] = Field(
        default_factory=list
    )

    session_memory: dict = Field(
        default_factory=dict
    )

    plan: Optional[AgentPlan] = None

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

    def remember(
            self,
            key: str,
            value: Any,
    ) -> None:
        self.session_memory[key] = value

    def recall(
            self,
            key: str,
    ) -> Optional[Any]:
        return self.session_memory.get(key)

    def set_plan(
            self,
            plan: AgentPlan,
    ) -> None:
        self.plan = plan

    def complete_plan_step(
            self,
            step_index: int,
    ) -> None:
        if self.plan is None:
            return

        if (
                step_index < 0
                or step_index >= len(
            self.plan.steps
        )
        ):
            raise IndexError(
                f"Invalid plan step index: "
                f"{step_index}"
            )

        self.plan.steps[
            step_index
        ].completed = True

    def get_recent_messages(
            self,
            max_user_turns: int,
    ) -> List[Any]:
        if max_user_turns <= 0:
            return []

        user_turn_count = 0
        start_index = 0

        for index in range(
                len(self.messages) - 1,
                -1,
                -1,
        ):
            message = self.messages[index]

            if isinstance(
                    message,
                    dict,
            ):
                role = message.get(
                    "role"
                )
            else:
                role = getattr(
                    message,
                    "role",
                    None,
                )

            if role == "user":
                user_turn_count += 1

                if (
                        user_turn_count
                        == max_user_turns
                ):
                    start_index = index
                    break

        if (
                user_turn_count
                < max_user_turns
        ):
            start_index = 0

        return self.messages[
            start_index:
        ]

    def build_llm_messages(
            self,
            max_recent_turns: int = 5,
    ) -> List[Any]:
        messages = []

        if self.session_memory:
            memory_lines = []

            for key, value in (
                    self.session_memory.items()
            ):
                memory_lines.append(
                    f"- {key}: {value}"
                )

            memory_text = "\n".join(
                memory_lines
            )

            memory_message = {
                "role": "system",
                "content": (
                    "Stored memory about the user:\n"
                    f"{memory_text}\n\n"
                    "Use these stored facts when they "
                    "are relevant to the user's request. "
                    "Stored memory may also be used to "
                    "resolve references and construct "
                    "tool arguments. "
                    "Do not invent memory that is not "
                    "listed above."
                ),
            }

            messages.append(
                memory_message
            )

        recent_messages = (
            self.get_recent_messages(
                max_user_turns=(
                    max_recent_turns
                )
            )
        )

        messages.extend(
            recent_messages
        )

        return messages


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
