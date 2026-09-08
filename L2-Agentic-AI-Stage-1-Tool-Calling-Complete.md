# ✅ L2 — Agentic AI, Stage 1 — Tool Calling: COMPLETE

**Status:** Complete · **Snapshot date:** 8 September 2026 · **Next:** Stage 2 — Agent State & Memory

This document records the completed Stage 1 tutorial: what was built, how the parts work together, what the final tests demonstrated, and where learning continues. It is based on the implementation and test results recorded in [L2 — Agentic AI](https://chatgpt.com/c/6a95557d-da0c-83eb-a5b8-16b23a5ecc26). It is a tutorial progress snapshot, not a new code audit or test run.

## What was built

A working Python agent can receive a question, let an LLM request tools, execute those requests through a shared validation pipeline, return structured results to the model, and repeat until the model produces a final answer or the application reaches its iteration limit.

The completed system supports:

- **Multiple typed tools:** `calculator`, `get_weather`, and `get_time`, each with an input and output model.
- **LLM tool selection:** the model chooses tools and constructs arguments from the user’s request and the available schemas.
- **Pydantic input validation:** arguments are checked against each tool’s input model before execution.
- **Generic execution:** a registry resolves the tool name, then delegates to `ToolDefinition.execute(arguments)`.
- **Typed output validation:** tool results follow declared output models and are normalized for serialization.
- **Structured `ToolResult`:** a common envelope carries `success`, `data`, and `error`.
- **Partial-failure recovery:** a failed tool returns an error result while other requested work can still succeed.
- **A multi-iteration agent loop:** tool results return to the model, which can request further tools or answer.
- **Multiple calls in one model response:** the loop handles both individual and grouped tool requests.

## Architecture

There are two connected paths: exposing tools to the model and executing the model’s requests.

```text
Tool registry: ToolDefinition objects
    │
    └── to_llm_schema()
            └── input_model.model_json_schema()
                    │
                    ▼
User question ──► Agent loop ──► LLM receives messages + tool schemas
                    ▲                         │
                    │                         ├── Final answer ──► User
                    │                         │
                    │                         └── Tool request(s)
                    │                                  │
                    │                           Parse JSON arguments
                    │                                  │
                    │                           execute_tool(name, args)
                    │                                  │
                    │                           Look up ToolDefinition
                    │                                  │
                    │                           execute(arguments)
                    │                                  │
                    │                           Validate input model
                    │                                  │
                    │                           Call Python function
                    │                                  │
                    │                           Validate output model
                    │                                  │
                    │                           Normalize result data
                    │                                  │
                    └── Append serialized result ◄── ToolResult
                                                       ▲
                                                       │
                                        Execution/validation failure
                                        becomes a failure ToolResult
```

The application enforces an iteration limit around the loop. An unknown tool name is also represented as a failed result rather than dispatched to an arbitrary function.

### The shared tool contract

| Tool | Input model | Python function | Output model |
| --- | --- | --- | --- |
| Calculator | `CalculatorInput` | `calculator()` | `CalculatorResult` |
| Weather | `WeatherInput` | `get_weather()` | `WeatherResult` |
| Time | `TimeInput` | `get_time()` | `TimeResult` |

Each `ToolDefinition` holds `name`, `description`, `function`, `input_model`, and an optional `output_model`. All three current tools register an output model.

- `to_llm_schema()` describes the tool to the LLM using a schema generated from its input model.
- `execute(arguments)` validates inputs, calls the function, handles typed outputs, and creates the common result envelope.

For example, a successful calculation is returned in this serialized shape:

```json
{
  "success": true,
  "data": {"expression": "50 * 20", "result": 1000.0},
  "error": null
}
```

A failure uses the same envelope:

```json
{
  "success": false,
  "data": null,
  "error": "Unsupported city: Atlanta. Supported cities are: Berlin, London, New York, Tokyo."
}
```

The tool-specific output model defines the payload contract; `ToolResult` provides the common success/failure wrapper.

## The key mental model

> **The LLM decides what action to request; the application decides whether and how that action is actually executed.**

| LLM responsibilities | Application responsibilities |
| --- | --- |
| Interpret the user’s question | Define available tools and their schemas |
| Select a tool or tools | Resolve requests through the tool registry |
| Construct arguments | Parse and validate arguments |
| Interpret returned results | Execute Python functions and validate outputs |
| Decide whether to request another action | Handle failures and return structured results |
| Produce a final answer | Maintain the loop’s messages and enforce iteration limits |

The model requests execution; Python performs it. A schema describes expected arguments, while application-side validation checks the actual arguments received.

Authorization also belongs on the application side. Stage 1 establishes the execution boundary; it does not claim a production permission system. More extensive authorization, observability, retries, timeouts, cost controls, and security remain later production concerns.

## Final test outcomes

Both closing tests passed in the recorded tutorial session. To repeat them, run `python3 agent.py` from the tutorial’s `experiments/agentic_ai/stage1/` directory and enter the prompts below. Exact call order, iteration count, and final wording can vary.

### Test 1 — All three tools across successive iterations

**Prompt:**

```text
What time is it in Tokyo, what is the weather in Berlin, and what is 123 * 456?
```

| LLM iteration | Observed action | Recorded outcome |
| --- | --- | --- |
| 1 | `get_time({"city": "Tokyo"})` | Success: `01:49:21`, `Asia/Tokyo` |
| 2 | `get_weather({"city": "Berlin"})` | Success: `18.0°C`, sunny |
| 3 | `calculator({"expression": "123 * 456"})` | Success: `56088.0` |
| 4 | No tool call; final answer | Combined all three results |

The time and weather values above are historical tool outputs from the exercise, not current conditions.

**What this demonstrated:** the model can spread work across several iterations. The application successfully carries each result back into the next model interaction without a hard-coded time → weather → calculator route.

### Test 2 — Atlanta partial failure with grouped tool requests

**Prompt:**

```text
What time is it in Atlanta and what is 50 * 20?
```

```text
LLM iteration 1
├── get_time(Atlanta)
│   └── FAILURE: unsupported city
└── calculator(50 * 20)
    └── SUCCESS: 1000.0

LLM iteration 2
└── Final answer: explains the time limitation and reports 1000
```

The time tool reported that it supports Berlin, London, New York, and Tokyo. The agent accurately explained that Atlanta was unavailable and still returned the successful calculation.

**What this demonstrated:** one tool failure did not destroy the whole task. Recovery here means continuing with the successful work and reporting the limitation; the failed time lookup itself was not repaired.

### Sequential versus parallel tool calls

The first test used one tool request per model iteration. The second requested two tools in the same model response, often described as **parallel tool calling**.

These observations establish support for both sequential and grouped requests. They do **not** establish concurrent Python execution: requesting multiple tools at once and running their functions concurrently are separate implementation choices.

### Supporting checks

Earlier recorded checks confirmed that all three tools returned the common success envelope with typed data. After the `ToolDefinition` refactor, schema generation exposed all three tools and a direct calculator execution returned `300.0` for `25 * 12`.

## Completed Stage 1 concepts and steps

This checklist consolidates the completed learning sequence; it does not recreate every original substep number.

- [x] Understand what a tool is and how a model requests it.
- [x] Build the first calculator tool.
- [x] Describe tools with names, descriptions, and argument schemas.
- [x] Connect the LLM to a real tool-calling interaction.
- [x] Parse JSON tool arguments and dispatch requests through a registry.
- [x] Add weather and time tools.
- [x] Let the LLM select among multiple tools.
- [x] Handle multiple tool calls in one model response.
- [x] Build a generic agent loop with repeated model/tool interactions.
- [x] Return tool results to the model and generate a final answer.
- [x] Enforce iteration limits.
- [x] Handle tool errors and unknown tool names through structured results.
- [x] Validate tool inputs with Pydantic.
- [x] Generate JSON schemas from input models.
- [x] Standardize results with `ToolResult(success, data, error)`.
- [x] Complete **Step 23 — Typed Tool Outputs** for all three tools.
- [x] Complete **Step 24 — ToolDefinition abstraction**: create the model, replace registry dictionaries, move execution into `execute()`, and move schema generation into `to_llm_schema()`.
- [x] Complete **Step 25 — Final End-to-End Stage 1 Agent Test**.
- [x] Pass the closing Atlanta partial-failure test.

## Current project files

Recorded tutorial directory: `experiments/agentic_ai/stage1/`.

| File | Responsibility at Stage 1 completion |
| --- | --- |
| `agent.py` | User interaction, LLM calls, message history within a run, tool-call processing, iteration limits, and final answers. |
| `tools.py` | Tool registration, the list of generated LLM schemas, and generic `execute_tool()` lookup/delegation. |
| `schemas.py` | Pydantic input/output models, `ToolResult`, and `ToolDefinition` with schema generation and execution behavior. |
| `calculator.py` | Calculator implementation returning `CalculatorResult`. |
| `weather.py` | Weather tool implementation returning `WeatherResult`. |
| `clock.py` | Supported-city timezone mapping and local-time lookup returning `TimeResult`. |

## Intentional learning-code limitation

**The calculator still uses `eval(expression)`. This is intentionally unsafe learning code, and replacing it was explicitly deferred.**

Validating `expression` as a string does not make evaluating it safe. This implementation is not suitable for untrusted expressions or production use. A constrained arithmetic parser/evaluator is a future replacement; calculator hardening is not part of the completed Stage 1 work.

## Transition to Stage 2 — Agent State & Memory

Stage 1 maintains enough message context to reason and use tools **inside one run**. In the recorded design, a new invocation starts with a new message list; after `run_agent()` finishes, that state is not retained for a later run.

For example, telling the agent “My name is Marko” in one invocation does not make that fact available when a separate invocation asks “What is my name?”

Stage 2 will distinguish:

- **Conversation history:** the messages exchanged with the model.
- **Agent state:** the information the application tracks while carrying out work.
- **Short-term memory:** context retained for ongoing interaction.
- **Long-term memory:** information retained for future interactions.

The next tutorial begins with **Step 1 — What actually is agent state?** The plan is to implement these ideas directly before introducing framework abstractions, following the same learning approach used for tools.

| L2 stage | Status at this snapshot |
| --- | --- |
| 1 — Tool Calling | **Complete** |
| 2 — Agent State & Memory | **Next; not started** |
| 3 — Planning & Multi-Step Agents | Upcoming |
| 4 — RAG / Knowledge Agents | Upcoming |
| 5 — Multi-Agent Systems | Upcoming |
| 6 — Integrated Agent Project | Upcoming |

**Resume point:** Stage 1 is closed. Begin Stage 2 with the distinction between transient messages, agent state, and memory.
