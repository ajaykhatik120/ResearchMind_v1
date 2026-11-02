
from core.vector_store import create_vector_store

from retrieval.hybrid import HybridRetriever
from retrieval.reranker import DocumentReranker

from agents.tools import build_document_search_tool
from tests.sample_data import documents


def main():

    # 1. Create vector database
    db = create_vector_store(
        documents,
        "document_tool_test_v1"
    )

    # 2. Initialize hybrid retriever
    hybrid = HybridRetriever(
        documents,
        db
    )

    # 3. Initialize reranker
    reranker = DocumentReranker()

    # 4. Build document search tool
    search_tool = build_document_search_tool(
        hybrid_retriever=hybrid,
        reranker=reranker
    )

    print("Tool name:", search_tool.name)

    # 5. Execute tool
    query = "How do gradients become very small?"

    result = search_tool.invoke({
        "query": query
    })

    print("\nDOCUMENT SEARCH RESULTS:")
    print(result)


if __name__ == "__main__":
    main()
