
from typing import Optional

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core.llm import get_llm


# ==================================================
# 1. SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are ResearchMind, an AI research assistant.

Rules:
1. Use calculator for mathematical calculations.
2. Use statistics_analyzer for statistical operations.
3. Use document_search for questions about uploaded documents.
4. Use web_search for current or external information.
5. Never claim that you searched a source unless you used its tool.
6. Clearly distinguish document evidence from web evidence.
7. If evidence is insufficient, explicitly say so.
8. Give concise, accurate answers.
"""


# ==================================================
# 2. CHECKPOINTER
# ==================================================

checkpointer = InMemorySaver()


# ==================================================
# 3. ROUTING FUNCTIONS
# ==================================================

def route_after_llm(state: MessagesState):
    """
    Check whether the LLM requested tool execution.
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "approval"

    return "end"


def route_after_approval(state):
    """
    Route according to the human decision.
    """

    if state.get("approved", False):
        return "tools"

    return "llm"


# ==================================================
# 4. BUILD AGENT GRAPH
# ==================================================

def build_agent_graph(tools):

    llm_with_tools = get_llm().bind_tools(tools)

    # ----------------------------------------------
    # LLM NODE
    # ----------------------------------------------

    def call_llm(state: MessagesState):

        model_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]

        response = llm_with_tools.invoke(
            model_messages
        )

        return {
            "messages": [response]
        }

    # ----------------------------------------------
    # HUMAN APPROVAL NODE
    # ----------------------------------------------

    def human_approval(state: MessagesState):

        last_message = state["messages"][-1]

        tool_calls = getattr(
            last_message,
            "tool_calls",
            [],
        )

        # Prepare tool details for the user.
        approval_request = {
            "message": "ResearchMind wants to execute these tools.",
            "tool_calls": [
                {
                    "name": call["name"],
                    "arguments": call.get("args", {}),
                    "id": call.get("id"),
                }
                for call in tool_calls
            ],
        }

        # Pause graph execution here.
        decision = interrupt(approval_request)

        approved = (
            decision.get("approved", False)
            if isinstance(decision, dict)
            else bool(decision)
        )

        # If rejected, return a ToolMessage for
        # each pending tool call. This allows the
        # LLM to continue without executing them.
        if not approved:

            rejection_messages = []

            for call in tool_calls:

                rejection_messages.append(
                    ToolMessage(
                        content=(
                            "The user rejected this tool execution. "
                            "Do not claim that the tool was executed. "
                            "Continue answering using available information."
                        ),
                        tool_call_id=call["id"],
                    )
                )

            return {
                "approved": False,
                "messages": rejection_messages,
            }

        return {
            "approved": True
        }

    # ----------------------------------------------
    # BUILD STATE GRAPH
    # ----------------------------------------------

    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_llm)

    builder.add_node(
        "approval",
        human_approval,
    )

    builder.add_node(
        "tools",
        ToolNode(tools),
    )

    # Starting point.
    builder.add_edge(
        START,
        "llm",
    )

    # LLM decides whether to finish or request approval.
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "approval": "approval",
            "end": END,
        },
    )

    # Human decision controls tool execution.
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tools": "tools",
            "llm": "llm",
        },
    )

    # After execution, return to LLM.
    builder.add_edge(
        "tools",
        "llm",
    )

    # Compile with checkpointing.
    return builder.compile(
        checkpointer=checkpointer,
    )


# ==================================================
# 5. GRAPH CONFIGURATION
# ==================================================

def get_graph_config(thread_id):

    if not thread_id:
        raise ValueError("thread_id is required.")

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 15,
    }


# ==================================================
# 6. RUN AGENT
# ==================================================

def run_agent_graph(
    question: str,
    tools: list,
    thread_id: str,
):

    agent = build_agent_graph(tools)

    config = get_graph_config(thread_id)

    # Count messages already stored in this thread.
    previous_state = agent.get_state(config)

    previous_count = len(
        previous_state.values.get("messages", [])
    )

    # Start graph execution.
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config=config,
    )

    # Detect whether graph is waiting for approval.
    interruptions = result.get("__interrupt__", [])

    pending_request = (
        interruptions[0].value
        if interruptions
        else None
    )

    all_messages = result.get("messages", [])

    return {
        "messages": all_messages,
        "new_messages": all_messages[previous_count:],
        "needs_approval": bool(interruptions),
        "approval_request": pending_request,
    }


# ==================================================
# 7. RESUME AGENT AFTER HUMAN DECISION
# ==================================================

def resume_agent_graph(
    tools: list,
    thread_id: str,
    approved: bool,
):

    agent = build_agent_graph(tools)

    config = get_graph_config(thread_id)

    # Get the number of messages before resuming.
    previous_state = agent.get_state(config)

    previous_count = len(
        previous_state.values.get("messages", [])
    )

    # Resume interrupted execution.
    result = agent.invoke(
        Command(
            resume={
                "approved": approved
            }
        ),
        config=config,
    )

    all_messages = result.get("messages", [])

    # Detect another pending approval.
    interruptions = result.get("__interrupt__", [])

    pending_request = (
        interruptions[0].value
        if interruptions
        else None
    )

    return {
        "messages": all_messages,

        "new_messages": all_messages[previous_count:],

        "needs_approval": bool(interruptions),

        "approval_request": pending_request,
    }