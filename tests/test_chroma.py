from langchain_core.documents import Document
from core.vector_store import create_vector_store

documents = [
    Document(page_content="A dog is playing in the garden."),
    Document(page_content="A puppy is running outside."),
    Document(page_content="Stock prices are increasing.")
]

# Create vector database
db = create_vector_store(documents)

# Search by meaning
results = db.similarity_search(
    "Tell me about a dog running.",
    k=2
)

for result in results:
    print(result.page_content)