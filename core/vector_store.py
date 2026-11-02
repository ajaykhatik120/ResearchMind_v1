def create_vector_store(documents, collection_name):
    """
    Create a Chroma vector store.
    Embedding dependencies are loaded only when needed.
    """

    from langchain_chroma import Chroma
    from core.embeddings import get_embeddings

    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db",
    )