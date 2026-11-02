from core.llm import get_llm

llm = get_llm()

response = llm.invoke(
    "Explain what an LLM is in three sentences."
)

# Inspect the response
print("\n--- RESPONSE ---")
print(response)

print("\n--- RESPONSE CONTENT ---")
print(response.content)

print("\n--- RESPONSE TYPE ---")
print(type(response))

print("\n--- MESSAGE ROLE ---")
print(response.type)

print("\n--- TOKEN USAGE ---")
print(response.usage_metadata)