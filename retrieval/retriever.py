
def retrieve_documents(
    vector_store,
    query,
    search_type="similarity",
    k=5,
    fetch_k=20,
    lambda_mult=0.8,
    filter=None
):
    """
    Retrieve documents using similarity or MMR,
    optionally restricting results using metadata.
    """

    search_kwargs = {
        "k": k
    }

    # Apply metadata filter when provided
    if filter is not None:
        search_kwargs["filter"] = filter

    if search_type == "similarity":

        results = vector_store.similarity_search(
            query,
            **search_kwargs
        )

    elif search_type == "mmr":

        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

        results = vector_store.max_marginal_relevance_search(
            query,
            **search_kwargs
        )

    else:
        raise ValueError(
            f"Unsupported search type: {search_type}"
        )

    return results
