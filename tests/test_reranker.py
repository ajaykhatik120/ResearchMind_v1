
from core.vector_store import create_vector_store
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import DocumentReranker
from tests.sample_data import documents


def main():

    db = create_vector_store(
        documents,
        "reranker_test_v1"
    )

    hybrid = HybridRetriever(
        documents,
        db
    )

    reranker = DocumentReranker()

    query = "How do gradients become very small?"

    candidates = hybrid.search(
        query,
        k=10
    )

    results = reranker.rerank(
        query,
        candidates,
        top_n=3
    )

    print("\nRERANKED RESULTS:")

    for i, result in enumerate(results, 1):

        doc = result["document"]
        score = result["score"]

        print(f"\nRank: {i}")
        print(f"Category: {doc.metadata['category']}")
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")


if __name__ == "__main__":
    main()
