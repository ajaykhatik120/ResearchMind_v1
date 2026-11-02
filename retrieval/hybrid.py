from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents, vector_store):

        self.documents = documents
        self.vector_store = vector_store

        # Prepare documents for BM25
        tokenized_documents = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def dense_search(self, query, k=5):

        return self.vector_store.similarity_search(
            query,
            k=k
        )

    def sparse_search(self, query, k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            self.documents[i]
            for i in ranked_indices[:k]
        ]

    def reciprocal_rank_fusion(
        self,
        dense_results,
        sparse_results,
        k=60
    ):

        fused_scores = {}
        document_map = {}

        result_lists = [
            dense_results,
            sparse_results
        ]

        for result_list in result_lists:

            for rank, doc in enumerate(result_list, 1):

                doc_id = id(doc)

                document_map[doc_id] = doc

                fused_scores[doc_id] = (
                    fused_scores.get(doc_id, 0)
                    + 1 / (k + rank)
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

        dense_results = self.dense_search(query, k=10)

        sparse_results = self.sparse_search(query, k=10)

        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )

        return fused_results[:k]