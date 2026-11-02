import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    if not model_name:
        raise ValueError("GROQ_MODEL is missing from .env")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=1000,
        reasoning_effort="low",
        reasoning_format="hidden"
    )

