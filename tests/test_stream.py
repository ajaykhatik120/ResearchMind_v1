from core.llm import get_llm

llm = get_llm()

for chunk in llm.stream(
    "Explain neural networks in 2 sentences."
):
    print(chunk.content, end="", flush=True)