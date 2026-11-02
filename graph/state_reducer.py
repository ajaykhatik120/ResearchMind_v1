
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


# 1. Define state with a reducer
class ResearchState(TypedDict):
    question: str
    results: Annotated[list, operator.add]


# 2. First node
def retrieve_documents(state: ResearchState):
    print("Retrieving documents...")

    return {
        "results": ["Document A", "Document B"]
    }


# 3. Second node
def search_web(state: ResearchState):
    print("Searching the web...")

    return {
        "results": ["Web Result A", "Web Result B"]
    }


# 4. Build graph
builder = StateGraph(ResearchState)

builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("search_web", search_web)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "search_web")
builder.add_edge("search_web", END)

# 5. Compile
research_graph = builder.compile()


# 6. Execute
if __name__ == "__main__":
    result = research_graph.invoke({
        "question": "Explain Transformers",
        "results": []
    })

    print("\nFinal results:")
    print(result["results"])
