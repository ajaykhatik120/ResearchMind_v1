from core.llm import get_llm

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

# 1. Connect to our LLM
llm = get_llm()

# 2. Define the assistant's role
# system_message = SystemMessage(
#     content="""
#     You are ResearchMind, an intelligent research assistant.

#     Your responsibilities:
#     - Explain technical concepts clearly.
#     - Use logical reasoning.
#     - Avoid inventing facts.
#     - Structure answers with headings and examples.
#     """
# )

system_message = SystemMessage(
    content="Answer every question in exactly two sentences."
)

# 3. Define the user's question
# human_message = HumanMessage(
#     content="Explain how a Transformer works."
# )

human_message =HumanMessage(
    content="What is overfitting in machine learning?"
)

# 4. Send both messages to the model
response = llm.invoke([
    system_message,
    human_message
])

# 5. Display the answer
print(response.content)