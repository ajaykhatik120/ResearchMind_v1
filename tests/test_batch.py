from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in maximum 2 sentences."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

questions = [
    {"question": "What is machine learning?"},
    {"question": "What is deep learning?"},
    {"question": "What is overfitting?"}
]

results = chain.batch(questions)

for i, result in enumerate(results, 1):
    print(f"\nQuestion {i}:")
    print(result)