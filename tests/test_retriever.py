
from core.vector_store import create_vector_store
from langchain_core.documents import Document
from retrieval.retriever import retrieve_documents
from langchain_core.documents import Document

from tests.sample_data import documents


db = create_vector_store(
    documents,
    "retriever_metadata_v1"
)

# Experiment 1: Search without filtering
results = retrieve_documents(
    db,
    "What is portfolio diversification?",
    k=5
)

print("\n--- WITHOUT FILTER ---")

for doc in results:
    print(doc.metadata["category"], ":", doc.page_content)


# Experiment 2: Search only finance documents
results = retrieve_documents(
    db,
    "What is portfolio diversification?",
    k=5,
    filter={"category": "finance"}
)

print("\n--- FINANCE FILTER ---")

for doc in results:
    print(doc.metadata["category"], ":", doc.page_content)


# Experiment 3: Search only deep learning documents
results = retrieve_documents(
    db,
    "How does backpropagation work?",
    k=5,
    filter={"category": "deep_learning"}
)

print("\n--- DEEP LEARNING FILTER ---")

for doc in results:
    print(doc.metadata["category"], ":", doc.page_content)



