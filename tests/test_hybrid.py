
import hashlib
import re

from rank_bm25 import BM25Okapi

from core.vector_store import create_vector_store
from retrieval.hybrid import HybridRetriever
from tests.sample_data import documents



class HybridRetriever:

    def __init__(self, documents, vector_store):
        self.documents = documents
        self.vector_store = vector_store

        tokenized_documents = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def tokenize(self, text):
        return re.findall(r"\b[\w-]+\b", text.lower())

    def get_document_id(self, doc):
        content = doc.page_content.strip().lower()
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def dense_search(self, query, k=5):
        return self.vector_store.similarity_search(
            query, k=k
        )

    def sparse_search(self, query, k=5):
        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = [
            self.documents[i]
            for i in ranked_indices
            if scores[i] > 0
        ]

        return results[:k]

    def reciprocal_rank_fusion(
        self, dense_results, sparse_results,
        rrf_constant=60
    ):
        fused_scores = {}
        document_map = {}

        for result_list in [dense_results, sparse_results]:

            # Prevent duplicate documents within one result list.
            seen = set()

            for rank, doc in enumerate(result_list, 1):

                doc_id = self.get_document_id(doc)

                if doc_id in seen:
                    continue

                seen.add(doc_id)
                document_map[doc_id] = doc

                fused_scores[doc_id] = (
                    fused_scores.get(doc_id, 0)
                    + 1 / (rrf_constant + rank)
                )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )

        return [
            document_map[doc_id]
            for doc_id in ranked_ids
        ]

    def search(self, query, k=5):

        dense_results = self.dense_search(
            query, k=10
        )

        sparse_results = self.sparse_search(
            query, k=10
        )

        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )

        return fused_results[:k]




def main():
    db = create_vector_store(
        documents,
        "hybrid_test_v4"
    )

    retriever = HybridRetriever(documents, db)

    queries = [
        "What is ResNet-50?",
        "How do gradients become very small?",
        "Explain portfolio diversification."
    ]

    for query in queries:
        print("\nQUERY:", query)

        results = retriever.search(query, k=5)

        for i, doc in enumerate(results, 1):
            print(
                f"{i}. [{doc.metadata['category']}] "
                f"{doc.page_content}"
            )


if __name__ == "__main__":
    main()