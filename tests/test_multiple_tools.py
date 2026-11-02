
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from core.llm import get_llm

from agents.tools import (
    calculator,
    statistics_analyzer
)

def main():

    llm = get_llm()

    # Give the model two tools
    tools = [
        calculator,
        statistics_analyzer
    ]

    llm_with_tools = llm.bind_tools(tools)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    messages = [
        SystemMessage(
            content="""
            You are ResearchMind, an analytical assistant.

            Available tools:
            - calculator: for arithmetic calculations.
            - statistics_analyzer: for statistical analysis
              of numerical datasets.

            Always use the appropriate tool when a calculation
            or statistical analysis is requested.
            """
        ),
        HumanMessage(
            content=(
                "Calculate the mean and population standard "
                "deviation of these values: 10, 20, 30, 40, 50."
            )
        )
    ]

    # Initial model call
    response = llm_with_tools.invoke(messages)

    iteration = 0

    # Tool execution loop
    while response.tool_calls:

        iteration += 1

        if iteration > 5:
            raise RuntimeError(
                "Too many tool-calling iterations."
            )

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            arguments = tool_call["args"]

            print("\nSelected tool:", tool_name)
            print("Arguments:", arguments)

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                result = f"Unknown tool: {tool_name}"
            else:
                result = selected_tool.invoke(arguments)

            print("Tool result:", result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Send tool results back to the LLM
        response = llm_with_tools.invoke(messages)

    print("\nFINAL ANSWER:")
    print(response.content)


if __name__ == "__main__":
    main()
