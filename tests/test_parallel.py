from core.llm import get_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = get_llm()
parser = StrOutputParser()

def create_chain(instruction):
    prompt = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "{topic}")
    ])
    return prompt | llm | parser


parallel_chain = RunnableParallel({
    "summary": create_chain(
        "Summarize in 2 sentences."
    ),
    "keywords": create_chain(
        "Extract 5 important keywords."
    ),
    "questions": create_chain(
        "Generate 2 research questions."
    )
})

result = parallel_chain.invoke({
    "topic": "Explain the applications of deep learning in healthcare."
})

for task, output in result.items():
    print(f"\n{task.upper()}:")
    print(output)


#tests/test_batch.py