from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage


# 1. Define a node
def chatbot(state: MessagesState):

    # Read the latest user message
    user_message = state["messages"][-1].content

    # Generate a simple response
    response = f"You asked: {user_message}"

    # Return a new AI message
    return {
        "messages": [
            AIMessage(content=response)
        ]
    }


# 2. Build the graph
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 3. Compile
chatbot_graph = builder.compile()


# 4. Execute
if __name__ == "__main__":

    result = chatbot_graph.invoke({
        "messages": [
            HumanMessage(content="What is a Transformer?")
        ]
    })

    for message in result["messages"]:
        print(f"{message.type}: {message.content}")



# Second interaction using the previous conversation
result = chatbot_graph.invoke({
    "messages": result["messages"] + [
        HumanMessage(content="Explain its biggest advantage.")
    ]
})

print("\nUpdated conversation:")

for message in result["messages"]:
    print(f"{message.type}: {message.content}")