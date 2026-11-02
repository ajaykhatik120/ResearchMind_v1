
from langchain_core.tools import tool
import math
import statistics
from typing import Literal
from ddgs import DDGS

@tool
def calculator(a: float, b: float, operation: str) -> str:
    """
    Perform basic mathematical calculations.

    Args:
        a: First number.
        b: Second number.
        operation: One of add, subtract, multiply, divide.
    """

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        result = a * b

    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero is not allowed."

        result = a / b

    else:
        return "Error: Unsupported operation."

    return str(result)




@tool
def statistics_analyzer(
    values: list[float],
    operation: Literal[
        "mean",
        "median",
        "standard_deviation",
        "minimum",
        "maximum"
    ]
) -> str:
    """
    Analyze a numerical dataset.

    Args:
        values: List of numerical observations.
        operation: Statistical operation to perform.
        
        standard_deviation calculates population standard deviation.
    """

    if not values:
        return "Error: Dataset cannot be empty."

    if not all(math.isfinite(x) for x in values):
        return "Error: All values must be finite numbers."

    if operation == "mean":
        result = statistics.mean(values)

    elif operation == "median":
        result = statistics.median(values)

    elif operation == "standard_deviation":
        result = statistics.pstdev(values)

    elif operation == "minimum":
        result = min(values)

    elif operation == "maximum":
        result = max(values)

    else:
        return "Error: Unsupported statistical operation."

    return f"{operation}: {result:.6f}"




from langchain_core.tools import tool


def build_document_search_tool(
    hybrid_retriever,
    reranker,
    candidate_k=10,
    top_n=3
):
    """
    Build a document search tool using the active
    document collection's retriever and reranker.
    """

    @tool
    def document_search(query: str) -> str:
        """
        Search the user's uploaded research documents.

        Use this tool when a question requires information
        from uploaded PDFs, research papers, or notes.

        Args:
            query: The question or topic to search for.
        """

        # Step 1: Hybrid retrieval + RRF
        candidates = hybrid_retriever.search(
            query,
            k=candidate_k
        )

        if not candidates:
            return "No matching information found in the documents."

        # Step 2: Cross-encoder reranking
        results = reranker.rerank(
            query,
            candidates,
            top_n=top_n
        )

        results = [
            result
            for result in results
            if result["score"] > 0
            ]

        if not results:
            return "No relevant information found in the documents."

        # Step 3: Format retrieved information
        output = []

        for i, result in enumerate(results, 1):

            doc = result["document"]
            score = result["score"]

            source = doc.metadata.get(
                "source", "Unknown"
            )

            page = doc.metadata.get("page")

            if page is not None:
                try:
                    page = int(page) + 1
                except (ValueError, TypeError):
                    page = "Unknown"
            else:
                page = "Unknown"

            output.append(
                f"[Source {i}]\n"
                f"File: {source}\n"
                f"Page: {page}\n"
                f"Relevance score: {score:.4f}\n"
                f"Content: {doc.page_content}"
            )

        return "\n\n".join(output)

    return document_search



@tool
def web_search(query: str) -> str:
    """
    Search the web for information.

    Use this tool when the user asks for current information,
    external research, recent developments, or information
    not available in uploaded documents.

    Args:
        query: The topic or question to search for.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        search_results = DDGS(timeout=10).text(
            query,
            region="in-en",
            max_results=5,
            backend="auto"
        )

        if not search_results:
            return "No web search results found."

        formatted_results = []

        for i, result in enumerate(search_results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("href", "")
            snippet = result.get("body", "No description available.")

            formatted_results.append(
                f"[Web Source {i}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Information: {snippet}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Web search failed: {type(error).__name__}: {error}"