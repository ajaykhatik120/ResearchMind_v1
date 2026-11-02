
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from core.llm import get_llm
from agents.tools import calculator


def main():

    # 1. Load the LLM
    llm = get_llm()

    # 2. Give the LLM access to our calculator
    tools = [calculator]

    llm_with_tools = llm.bind_tools(tools)

    # 3. Create a tool lookup
    tool_map = {
        tool.name: tool
        for tool in tools
    }

    # 4. Start conversation
    messages = [
        SystemMessage(
            content=(
                "You are ResearchMind. "
                "Use the calculator tool for mathematical "
                "calculations. Do not calculate manually."
            )
        ),
        HumanMessage(
            content="Calculate 25678 multiplied by 9432."
        )
    ]

    # 5. Ask the LLM
    response = llm_with_tools.invoke(messages)

    # 6. Execute requested tools
    iteration = 0

    while response.tool_calls:

        iteration += 1

        if iteration > 5:
            raise RuntimeError("Too many tool-calling iterations.")

        # Preserve the assistant's tool-call message
        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            arguments = tool_call["args"]

            print("\nTool selected:", tool_name)
            print("Arguments:", arguments)

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                result = f"Unknown tool: {tool_name}"
            else:
                result = selected_tool.invoke(arguments)

            print("Tool result:", result)

            # Send tool output back to the LLM
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        # 7. Let the LLM interpret the tool result
        response = llm_with_tools.invoke(messages)

    print("\nFINAL ANSWER:")
    print(response.content)


if __name__ == "__main__":
    main()
