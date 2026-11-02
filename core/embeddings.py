from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the embedding model only when first needed.
    Reuse the same instance afterward.
    """

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )