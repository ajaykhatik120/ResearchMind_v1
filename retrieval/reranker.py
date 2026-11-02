
from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(
            model_name,
            device="cpu"
        )

    
    def rerank(self, query, documents, top_n=3):

        if not documents:
            return []

        # Remove duplicate documents
        unique_documents = []
        seen = set()

        for doc in documents:
            doc_id = doc.page_content.strip().lower()

            if doc_id not in seen:
                seen.add(doc_id)
                unique_documents.append(doc)

        pairs = [
            (query, doc.page_content)
            for doc in unique_documents
        ]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(unique_documents, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        return [
            {
                "document": doc,
                "score": float(score)
            }
            for doc, score in ranked_results[:top_n]
        ]
