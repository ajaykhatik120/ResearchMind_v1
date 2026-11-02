import sys
import os
import tempfile
import uuid
from pathlib import Path


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# 2. STREAMLIT AND PROJECT IMPORTS
# ==================================================

import streamlit as st

from ingestion.chunker import chunk_documents
from ingestion.pdf_loader import load_pdf

from core.vector_store import create_vector_store

from retrieval.hybrid import HybridRetriever
from retrieval.reranker import DocumentReranker

from agents.tools import (
    calculator,
    statistics_analyzer,
    web_search,
    build_document_search_tool,
)


# ==================================================
# 3. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ResearchMind")

st.caption(
    "Your AI-powered research assistant"
)


# ==================================================
# 4. SESSION STATE INITIALIZATION
# ==================================================

if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = uuid.uuid4().hex


if "chat_history" not in st.session_state:

    st.session_state["chat_history"] = []


if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None

if "pending_messages" not in st.session_state:
    st.session_state["pending_messages"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None


# ==================================================
# 5. CACHED MODEL LOADING
# ==================================================

@st.cache_resource
def load_reranker():
    """
    Load the cross-encoder only once.

    Streamlit reuses the same model instance
    across application reruns.
    """

    return DocumentReranker()


# ==================================================
# 6. LAZY RERANKER
# ==================================================

class LazyReranker:
    """
    Delay cross-encoder initialization until
    document reranking is actually requested.
    """

    def rerank(self, query, documents, top_n=3):

        reranker = load_reranker()

        return reranker.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )


# ==================================================
# 7. HELPER FUNCTIONS
# ==================================================

def get_page_number(metadata):
    """
    Convert zero-based PDF page numbers
    into human-readable page numbers.
    """

    page = metadata.get("page")

    if page is None:
        return "Unknown"

    try:
        return int(page) + 1

    except (ValueError, TypeError):
        return "Unknown"


def format_message_content(content):
    """
    Convert LangChain message content into
    displayable text.
    """

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):

                text_parts.append(
                    block.get("text", "")
                )

            else:

                text_parts.append(str(block))

        return "\n".join(
            part for part in text_parts if part
        )

    return str(content)


# ==================================================
# 8. ADVANCED RETRIEVAL
# ==================================================

def retrieve_and_rerank(
    query,
    candidate_k=10,
    top_n=3,
):
    """
    Hybrid retrieval followed by cross-encoder reranking.
    """

    hybrid = st.session_state["hybrid_retriever"]

    # STEP 1: Hybrid retrieval

    candidates = hybrid.search(
        query,
        k=candidate_k,
    )

    if not candidates:

        return []

    # STEP 2: Reranking

    reranker = load_reranker()

    results = reranker.rerank(
        query=query,
        documents=candidates,
        top_n=top_n,
    )

    return results


# ==================================================
# 9. BUILD AGENT TOOLS
# ==================================================

def build_agent_tools():
    """
    Build the tools available to ResearchMind.

    Always available:
        - Calculator
        - Statistics Analyzer
        - Web Search

    Available after PDF processing:
        - Document Search
    """

    tools = [
        calculator,
        statistics_analyzer,
        web_search,
    ]

    # Add document search only if a PDF
    # has been successfully processed.

    if (
        "hybrid_retriever" in st.session_state
        and "chunks" in st.session_state
    ):

        document_tool = build_document_search_tool(
            hybrid_retriever=st.session_state[
                "hybrid_retriever"
            ],

            # Do not load the model here.
            reranker=LazyReranker(),

            candidate_k=10,
            top_n=3,
        )

        tools.append(document_tool)

    return tools


# ==================================================
# 10. LANGGRAPH AGENT EXECUTION
# ==================================================

def run_agent(question, thread_id):

    # Lazy import.
    from graph.agent_graph import run_agent_graph

    tools = build_agent_tools()

    result = run_agent_graph(
        question=question,
        tools=tools,
        thread_id=thread_id,
    )

    return result


def resume_agent(thread_id, approved):

    # Lazy import.
    from graph.agent_graph import resume_agent_graph

    tools = build_agent_tools()

    result = resume_agent_graph(
        tools=tools,
        thread_id=thread_id,
        approved=approved,
    )

    return result



def extract_final_answer(messages):

    for message in reversed(messages):

        if message.type != "ai":
            continue

        if getattr(message, "tool_calls", None):
            continue

        if message.content:

            return format_message_content(
                message.content
            )

    return "I couldn't generate a final answer."


def extract_tool_logs(messages):

    tool_logs = []

    tool_logs_by_id = {}

    for message in messages:

        if message.type == "ai":

            for tool_call in (
                getattr(message, "tool_calls", []) or []
            ):

                log = {
                    "name": tool_call["name"],
                    "arguments": tool_call.get("args", {}),
                    "result": "",
                }

                tool_logs.append(log)

                call_id = tool_call.get("id")

                if call_id:
                    tool_logs_by_id[call_id] = log

        elif message.type == "tool":

            call_id = getattr(
                message,
                "tool_call_id",
                None,
            )

            matching_call = tool_logs_by_id.get(
                call_id
            )

            if matching_call:

                matching_call["result"] = (
                    format_message_content(
                        message.content
                    )[:4000]
                )

    return tool_logs


# ==================================================
# 11. PDF UPLOAD AND PROCESSING
# ==================================================

st.sidebar.header("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your research PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    if st.sidebar.button("Process PDF"):

        temp_path = None

        try:

            # --------------------------------------
            # STEP 1: SAVE TEMPORARY PDF
            # --------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_path = temp_file.name

            # --------------------------------------
            # STEP 2: PROCESS PDF
            # --------------------------------------

            with st.spinner(
                "Processing your research document..."
            ):

                # Extract pages.

                documents = load_pdf(temp_path)

                # Preserve source metadata.

                for document in documents:

                    document.metadata["source"] = (
                        uploaded_file.name
                    )

                # ----------------------------------
                # STEP 3: CHUNK DOCUMENTS
                # ----------------------------------

                chunks = chunk_documents(
                    documents
                )

                if not chunks:

                    raise ValueError(
                        "No text chunks were extracted "
                        "from this PDF."
                    )

                # ----------------------------------
                # STEP 4: CREATE CHROMA COLLECTION
                # ----------------------------------

                collection_name = (
                    f"researchmind_"
                    f"{uuid.uuid4().hex[:12]}"
                )

                vector_store = create_vector_store(
                    chunks,
                    collection_name,
                )

                # ----------------------------------
                # STEP 5: BUILD HYBRID RETRIEVER
                # ----------------------------------

                hybrid_retriever = HybridRetriever(
                    chunks,
                    vector_store,
                )

            # --------------------------------------
            # STEP 6: SAVE DOCUMENT RESOURCES
            # --------------------------------------

            st.session_state["documents"] = documents

            st.session_state["chunks"] = chunks

            st.session_state["vector_store"] = (
                vector_store
            )

            st.session_state["hybrid_retriever"] = (
                hybrid_retriever
            )

            st.session_state["collection_name"] = (
                collection_name
            )

            # --------------------------------------
            # STEP 7: RESET CONVERSATION
            # --------------------------------------

            st.session_state["chat_history"] = []

            # New thread prevents old document
            # conversation context from being reused.

            st.session_state["thread_id"] = (
                uuid.uuid4().hex
            )

            st.sidebar.success(
                f"Successfully processed "
                f"{len(documents)} pages and "
                f"{len(chunks)} chunks!"
            )

        except Exception as error:

            st.sidebar.error(
                f"PDF processing failed: "
                f"{type(error).__name__}: {error}"
            )

        finally:

            # Remove temporary file.

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)


# ==================================================
# 12. DOCUMENT STATUS
# ==================================================

if "chunks" in st.session_state:

    st.sidebar.divider()

    st.sidebar.success(
        "Document ready for research"
    )

    st.sidebar.metric(
        "Total chunks",
        len(st.session_state["chunks"]),
    )

    st.sidebar.caption(
        f"Collection: "
        f"{st.session_state['collection_name']}"
    )


# ==================================================
# 13. AGENT TOOL STATUS
# ==================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🛠️ Agent Tools"
)

tool_names = [
    "Calculator",
    "Statistics Analyzer",
    "Web Search",
]

if "hybrid_retriever" in st.session_state:

    tool_names.append(
        "Document Search"
    )

for tool_name in tool_names:

    st.sidebar.write(
        f"✅ {tool_name}"
    )


# ==================================================
# 14. ADVANCED RETRIEVAL SEARCH
# ==================================================

if "hybrid_retriever" in st.session_state:

    st.divider()

    st.subheader(
        "🔍 Advanced Retrieval Search"
    )

    st.caption(
        "Hybrid Retrieval → RRF → "
        "Cross-Encoder Reranking"
    )

    query = st.text_input(
        "Enter your search query",
        key="advanced_search_query",
    )

    if st.button(
        "Search",
        key="advanced_search_button",
    ):

        if query.strip():

            try:

                with st.spinner(
                    "Retrieving and reranking documents..."
                ):

                    results = retrieve_and_rerank(
                        query=query,
                        candidate_k=10,
                        top_n=3,
                    )

                if not results:

                    st.info(
                        "No matching documents were found."
                    )

                for i, result in enumerate(
                    results,
                    start=1,
                ):

                    doc = result["document"]

                    score = result["score"]

                    st.markdown(
                        f"### Result {i}"
                    )

                    st.caption(
                        f"Reranking score: {score:.4f}"
                    )

                    st.write(
                        doc.page_content
                    )

                    source = doc.metadata.get(
                        "source",
                        "Unknown",
                    )

                    page = get_page_number(
                        doc.metadata
                    )

                    st.caption(
                        f"Source: {source} | Page: {page}"
                    )

            except Exception as error:

                st.error(
                    f"Search failed: "
                    f"{type(error).__name__}: {error}"
                )


# ==================================================
# 15. DOCUMENT CHUNK INSPECTION
# ==================================================

if "chunks" in st.session_state:

    st.divider()

    st.subheader(
        "📚 Document Chunk Inspection"
    )

    chunks = st.session_state["chunks"]

    st.write(
        "Total chunks:",
        len(chunks),
    )

    selected_chunk = st.selectbox(
        "Select a chunk",
        range(len(chunks)),
    )

    chunk = chunks[selected_chunk]

    st.write("Metadata:")

    st.json(
        chunk.metadata
    )

    st.write("Chunk content:")

    st.text(
        chunk.page_content
    )


# ==================================================
# ==================================================
# 16. CHAT INTERFACE
# ==================================================

st.divider()

st.subheader("💬 Ask ResearchMind")

st.caption(
    "Research across uploaded documents, "
    "the web, mathematics and statistics."
)


# ==================================================
# 17. CLEAR CONVERSATION
# ==================================================

if st.session_state["chat_history"]:

    if st.button(
        "Clear conversation",
        key="clear_conversation",
    ):

        if st.session_state["pending_approval"] is not None:

            st.warning(
                "Resolve the pending approval before "
                "clearing this conversation."
            )

        else:

            st.session_state["chat_history"] = []

            st.session_state["thread_id"] = (
                uuid.uuid4().hex
            )

            st.rerun()


# ==================================================
# 18. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state["chat_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display tool logs for assistant messages.
        if message.get("tool_logs"):

            with st.expander(
                "🛠️ View tool execution details"
            ):

                for index, log in enumerate(
                    message["tool_logs"],
                    start=1,
                ):

                    st.markdown(
                        f"**{index}. {log['name']}**"
                    )

                    st.caption(
                        f"Arguments: {log['arguments']}"
                    )

                    if log["result"]:

                        st.text(log["result"])


# ==================================================
# 19. FINALIZE AGENT RESPONSE
# ==================================================

def finalize_agent_response():

    messages = st.session_state["pending_messages"]

    answer = extract_final_answer(messages)

    tool_logs = extract_tool_logs(messages)

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "tool_logs": tool_logs,
    })

    # Clear pending execution state.
    st.session_state["pending_approval"] = None

    st.session_state["pending_messages"] = []

    st.session_state["pending_question"] = None


# ==================================================
# 20. HUMAN APPROVAL INTERFACE
# ==================================================

if st.session_state["pending_approval"] is not None:

    approval = st.session_state["pending_approval"]

    st.divider()

    st.warning(
        "ResearchMind requires your approval "
        "before executing a tool."
    )

    st.subheader("🔐 Tool Execution Approval")

    st.write(
        approval.get(
            "message",
            "ResearchMind wants to execute tools.",
        )
    )

    # Display each requested tool.
    for index, tool_call in enumerate(
        approval.get("tool_calls", []),
        start=1,
    ):

        st.markdown(
            f"**Tool {index}: {tool_call['name']}**"
        )

        st.json(
            tool_call.get("arguments", {})
        )

    st.info(
        "Review the requested operation before approving."
    )

    col1, col2 = st.columns(2)

    # ----------------------------------------------
    # APPROVE
    # ----------------------------------------------

    with col1:

        approve_clicked = st.button(
            "✅ Approve and Execute",
            key="approve_tool",
            type="primary",
            use_container_width=True,
        )

    # ----------------------------------------------
    # REJECT
    # ----------------------------------------------

    with col2:

        reject_clicked = st.button(
            "❌ Reject",
            key="reject_tool",
            use_container_width=True,
        )

    # ----------------------------------------------
    # RESUME GRAPH
    # ----------------------------------------------

    if approve_clicked or reject_clicked:

        approved = approve_clicked

        try:

            with st.spinner(
                "Resuming ResearchMind..."
            ):

                result = resume_agent(
                    thread_id=st.session_state["thread_id"],
                    approved=approved,
                )

                # Accumulate messages across approval cycles.
                st.session_state["pending_messages"].extend(
                    result["new_messages"]
                )

                if result["needs_approval"]:

                    # Another tool requires approval.
                    st.session_state["pending_approval"] = (
                        result["approval_request"]
                    )

                else:

                    # Graph completed.
                    finalize_agent_response()

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to resume agent: "
                f"{type(error).__name__}: {error}"
            )


# ==================================================
# 21. NEW USER QUESTION
# ==================================================

if st.session_state["pending_approval"] is None:

    question = st.chat_input(
        "Ask ResearchMind anything...",
        key="researchmind_chat_input"
    )

    if question:

        # ------------------------------------------
        # DISPLAY USER MESSAGE
        # ------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state["chat_history"].append({
            "role": "user",
            "content": question,
        })

        st.session_state["pending_question"] = question

        # ------------------------------------------
        # START AGENT
        # ------------------------------------------

        try:

            with st.spinner(
                "ResearchMind is thinking..."
            ):

                result = run_agent(
                    question=question,
                    thread_id=st.session_state["thread_id"],
                )

                # Store newly generated messages.
                st.session_state["pending_messages"] = (
                    result["new_messages"]
                )

                if result["needs_approval"]:

                    # Pause and show approval UI.
                    st.session_state["pending_approval"] = (
                        result["approval_request"]
                    )

                else:

                    # No approval required.
                    finalize_agent_response()

            st.rerun()

        except Exception as error:

            error_message = (
                f"Unable to process your request: "
                f"{type(error).__name__}: {error}"
            )

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": error_message,
            })

            st.error(error_message)
            st.rerun()

