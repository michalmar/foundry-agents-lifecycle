# =============================================================================
# dispatch.py — Local execution of agent function tools
# =============================================================================
#
# WHY THIS EXISTS:
#   When a Foundry agent invokes a custom *function* tool (e.g. calculator),
#   the Responses API does NOT execute your code. It returns the call back to
#   the CLIENT and waits for a `function_call_output` before producing a final
#   text answer. Both the test harness (test_agent.py) and the evaluation gate
#   (run_evaluation.py) therefore need to:
#     1. Detect function_call items in the response output
#     2. Execute the named function locally
#     3. Submit the result back to continue the run
#
#   This module centralizes the registry + executor so both scripts share one
#   source of truth. Built-in tools (code_interpreter, file_search) run
#   server-side and never reach this dispatch path.
#
# TO REGISTER A NEW FUNCTION TOOL:
#   1. Add execute_<name>() to src/agent/tools/<name>.py
#   2. Export it via src/agent/tools/__init__.py
#   3. Add an entry to build_function_tool_registry() below
# =============================================================================

import json
from collections.abc import Callable

from .calculator import execute_calculator


def build_function_tool_registry() -> dict[str, Callable]:
    """Build a name -> executor map for all custom function tools."""
    return {"calculator": execute_calculator}


def execute_function_tool(name: str, arguments: str) -> str:
    """
    Execute a function tool by name and return its JSON-serialized output.

    Args:
        name: The function tool name the agent invoked.
        arguments: JSON string of arguments from the Responses API.

    Returns:
        JSON string with the tool result, or an {"error": ...} payload that is
        surfaced back to the agent so failures are visible rather than silent.
    """
    registry = build_function_tool_registry()
    if name not in registry:
        return json.dumps({"error": f"Unknown function tool: {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid arguments JSON: {e}"})
    try:
        result = registry[name](**args)
    except Exception as e:  # noqa: BLE001 — surface the real failure to the agent
        return json.dumps({"error": f"Tool execution failed: {e}"})
    return json.dumps(result)


def run_agent_turn(
    openai_client,
    agent_name: str,
    user_input: str,
    max_iterations: int = 5,
) -> tuple[str, list[dict], str]:
    """
    Send a single user turn to a Foundry agent via the Responses API and drive
    the function-tool loop until the agent produces a final text answer.

    Returns:
        (answer_text, tool_calls, status) where tool_calls is a list of
        {"name", "arguments", "output"} dicts for diagnostics/UI and status is
        the final response status string.
    """
    conversation = openai_client.conversations.create()
    agent_ref = {"agent_reference": {"name": agent_name, "type": "agent_reference"}}

    response = openai_client.responses.create(
        conversation=conversation.id,
        extra_body=agent_ref,
        input=user_input,
    )

    executed_calls: list[dict] = []

    for _ in range(max_iterations):
        function_calls = [
            item
            for item in (getattr(response, "output", None) or [])
            if getattr(item, "type", None) == "function_call"
        ]
        if not function_calls:
            break

        tool_outputs = []
        for fc in function_calls:
            name = getattr(fc, "name", "")
            arguments = getattr(fc, "arguments", "{}")
            call_id = getattr(fc, "call_id", None) or getattr(fc, "id", "")
            output = execute_function_tool(name, arguments)
            executed_calls.append({"name": name, "arguments": arguments, "output": output})
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body=agent_ref,
            input=tool_outputs,
        )

    answer = getattr(response, "output_text", "") or ""
    elapsed_status = getattr(response, "status", "completed")
    return answer, executed_calls, elapsed_status
