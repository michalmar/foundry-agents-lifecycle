#!/usr/bin/env python3
# =============================================================================
# test_agent.py — Talk to your deployed agent
# =============================================================================
#
# PURPOSE:
#   After deploying an agent, you want to SEE it work.
#   This script connects to your Foundry project, finds the deployed agent,
#   and lets you chat with it — either a single query or interactive mode.
#
# EXECUTION FLOW:
#   1. Load config   → Get agent name for the target environment
#   2. Auth          → DefaultAzureCredential (same as deploy_agent.py)
#   3. Find agent    → Search by name in the Foundry project
#   4. Create thread → Each conversation is a thread
#   5. Send query    → Post user message to the thread
#   6. Process run   → Agent thinks, may call tools, generates response
#   7. Display       → Show response + any tool calls made
#
# USAGE:
#   # Single query
#   python src/scripts/test_agent.py --env dev --query "What is 2 + 2?"
#
#   # Interactive chat
#   python src/scripts/test_agent.py --env dev
#
# WHY THIS MATTERS FOR CI/CD:
#   This is the verification step. After deploy_agent.py creates the agent,
#   this script proves it actually works. In a pipeline, you'd run a
#   predefined set of queries (see run_evaluation.py for that).
#
# =============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_agent(client: AIProjectClient, agent_name: str):
    """Find a deployed agent by name. Returns the agent object or None."""
    for agent in client.agents.list():
        if agent.name == agent_name:
            return agent
    return None


def send_query(openai_client, agent_name: str, query: str, conversation_id: str = None) -> dict:
    """
    Send a query to the agent via the Foundry Responses API and drive the
    function-tool loop until the agent produces a final text answer.

    Custom function tools (e.g. calculator) are NOT executed server-side — the
    Responses API hands the call back to us and waits for function_call_output
    before producing a final answer. We execute the tool locally via the shared
    dispatch helper and submit the result so the agent can finish its turn.

    SDK v2 pattern: conversations.create() + responses.create() with
    agent_reference, looping on function_call items.
    """
    from src.agent.tools import execute_function_tool

    start_time = time.time()

    # Create a conversation if none provided
    if not conversation_id:
        conversation = openai_client.conversations.create()
        conversation_id = conversation.id

    agent_ref = {"agent_reference": {"name": agent_name, "type": "agent_reference"}}

    response = openai_client.responses.create(
        conversation=conversation_id,
        extra_body=agent_ref,
        input=query,
    )

    # Drive the function-tool loop: execute any function calls locally and
    # submit their outputs until the agent stops requesting tools.
    tool_calls = []
    for _ in range(5):
        function_calls = [
            item for item in (getattr(response, "output", None) or [])
            if getattr(item, "type", None) == "function_call"
        ]
        if not function_calls:
            break

        tool_outputs = []
        for fc in function_calls:
            name = getattr(fc, "name", "unknown")
            arguments = getattr(fc, "arguments", "{}")
            call_id = getattr(fc, "call_id", None) or getattr(fc, "id", "")
            output = execute_function_tool(name, arguments)
            tool_calls.append({"name": name, "arguments": arguments, "output": output})
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

        response = openai_client.responses.create(
            conversation=conversation_id,
            extra_body=agent_ref,
            input=tool_outputs,
        )

    elapsed = time.time() - start_time

    # Extract final response text
    response_text = response.output_text if hasattr(response, "output_text") else str(response)

    return {
        "response": response_text,
        "tool_calls": tool_calls,
        "elapsed_seconds": round(elapsed, 1),
        "conversation_id": conversation_id,
        "status": getattr(response, "status", "completed"),
    }


def display_result(result: dict) -> None:
    """Pretty-print the agent's response."""
    # Show tool calls if any
    for tc in result["tool_calls"]:
        args = tc["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(args, dict):
            arg_str = ", ".join(f"{v}" for v in args.values())
            print(f"  🔧 Tool: {tc['name']}({arg_str})")
        else:
            print(f"  🔧 Tool: {tc['name']}({args})")
        if tc.get("output"):
            print(f"       → {tc['output']}")

    # Show response
    print(f"  🤖 {result['response']}")
    print(f"  ⏱️  {result['elapsed_seconds']}s | status: {result['status']}")


def interactive_chat(openai_client, agent_name: str) -> None:
    """Run an interactive chat loop with the agent."""
    print(f"\n  💬 Interactive chat with '{agent_name}'")
    print("  Type your message, or 'exit'/'quit' to stop.\n")

    # Create a single conversation for the whole chat
    conversation = openai_client.conversations.create()

    while True:
        try:
            query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("  👋 Goodbye!")
            break

        print()
        result = send_query(openai_client, agent_name, query, conversation.id)
        display_result(result)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Test a deployed Foundry agent by sending it queries",
        epilog="""
Examples:
  # Single query
  python src/scripts/test_agent.py --env dev --query "What is 25 times 4?"

  # Interactive chat
  python src/scripts/test_agent.py --env dev

  # Test prod agent
  python src/scripts/test_agent.py --env prod --query "What can you help me with?"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=["dev", "test", "prod"],
        help="Environment to test (determines which agent to find)",
    )
    parser.add_argument(
        "--query",
        help="Single query to send. If omitted, starts interactive chat.",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        print("  ❌ AZURE_AI_PROJECT_ENDPOINT not set.")
        print("  Set it in .env or as an environment variable.")
        sys.exit(1)

    # Load agent config to get the agent name
    from src.agent.agent_definition import load_agent_config

    config = load_agent_config(args.env, project_root)

    print(f"\n{'='*60}")
    print(f"  TEST AGENT — Environment: {args.env.upper()}")
    print(f"{'='*60}")
    print(f"  Looking for: {config.name}")
    print(f"  Project:     {endpoint[:60]}...")

    # Connect and find agent
    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)

    with client:
        agent = find_agent(client, config.name)
        if not agent:
            print(f"\n  ❌ Agent '{config.name}' not found in this project.")
            print("  Have you deployed it? Run:")
            print(f"    python src/scripts/deploy_agent.py --env {args.env}")
            sys.exit(1)

        # Get model from agent version details
        latest = agent.get("versions", {}).get("latest", {})
        defn = latest.get("definition", {})
        model = defn.get("model", "unknown")

        print(f"  🔍 Found agent: {agent.name} (ID: {agent.id})")
        print(f"  🤖 Model: {model}")

        # Get OpenAI client for the Responses API
        openai_client = client.get_openai_client()

        if args.query:
            # Single query mode
            print(f"\n  💬 Query: {args.query}\n")
            result = send_query(openai_client, agent.name, args.query)
            display_result(result)
            print()
        else:
            # Interactive mode
            interactive_chat(openai_client, agent.name)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    main()
