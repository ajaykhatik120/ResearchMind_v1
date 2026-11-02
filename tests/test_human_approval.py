
from graph.agent_graph import (
    run_agent_graph,
    resume_agent_graph,
)

from agents.tools import calculator


tools = [calculator]

thread_id = "hitl_test_001"


# STEP 1: Ask a question that requires a tool.

result = run_agent_graph(
    question="Calculate 125 multiplied by 16.",
    tools=tools,
    thread_id=thread_id,
)

print("\n--- INITIAL GRAPH EXECUTION ---")

print("Approval required:", result["needs_approval"])

print("Approval request:", result["approval_request"])


# STEP 2: Approve the tool.

if result["needs_approval"]:

    result = resume_agent_graph(
        tools=tools,
        thread_id=thread_id,
        approved=True,
    )

    print("\n--- AFTER APPROVAL ---")

    print("Approval required:", result["needs_approval"])

    for message in result["messages"][-3:]:

        print("\nMessage type:", message.type)

        print("Content:", message.content)


# STEP 3: Inspect the final answer.

print("\n--- FINAL STATE ---")

for message in result["messages"]:

    if message.type == "ai" and not message.tool_calls:

        if message.content:
            print(message.content)
