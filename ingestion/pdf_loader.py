def load_pdf(file_path: str):
    """
    Load a PDF only when the user requests processing.
    """

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)

    return loader.load()