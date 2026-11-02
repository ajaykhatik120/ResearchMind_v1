from core.llm import get_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = get_llm()

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert machine learning teacher."),
    ("human", "Explain {topic} from first principles.")
])

# Create output parser
parser = StrOutputParser()

# Build complete pipeline
chain = prompt | llm | parser

# Execute
response = chain.invoke({
    "topic": "Gradient Descent"
})

print(response)

print("\nOutput type:", type(response))