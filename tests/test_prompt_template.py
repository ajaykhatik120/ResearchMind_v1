from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

# 1. Connect to the LLM
llm = get_llm()

# 2. Create a reusable prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert {subject} teacher.

        Explain concepts from first principles.
        Use intuitive examples and simple language.
        Adapt your explanation to a {level} student.
        """
    ),
    (
        "human",
        "Explain {topic}."
    )
])

# 3. Supply values for the variables
prompt = prompt_template.invoke({
    "subject": "Machine Learning",
    "level": "beginner",
    "topic": "Random Forest"
})

# 4. Send the generated messages to the LLM
response = llm.invoke(prompt)

# 5. Display the result
print(response.content)