from agents.tools import web_search

query = "latest developments in transformer architecture"

result = web_search.invoke({"query": query})

print("\nWEB SEARCH RESULTS\n")
print(result)