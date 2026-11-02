from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class ResearchState(TypedDict):
    question: str
    processed_question: str
    answer: str


# 2. Define Node A: Process the question
def process_question(state: ResearchState):
    question = state["question"]

    processed = question.strip().lower()

    return {
        "processed_question": processed
    }


# 3. Define Node B: Generate a simple response
def generate_answer(state: ResearchState):
    question = state["processed_question"]

    answer = f"You asked: {question}"

    return {
        "answer": answer
    }


# 4. Build the graph
builder = StateGraph(ResearchState)

builder.add_node("process_question", process_question)
builder.add_node("generate_answer", generate_answer)

# 5. Connect the nodes
builder.add_edge(START, "process_question")
builder.add_edge("process_question", "generate_answer")
builder.add_edge("generate_answer", END)

# 6. Compile the graph
research_graph = builder.compile()


# 7. Execute the graph
if __name__ == "__main__":
    result = research_graph.invoke({
        "question": "  Explain Transformer Architecture  ",
        "processed_question": "",
        "answer": ""
    })

    print(result)
