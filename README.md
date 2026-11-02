
# ResearchMind

An AI-powered research assistant built using LangChain,
LangGraph, Groq, ChromaDB and Streamlit.

## Features

- PDF ingestion and document processing
- Semantic search using vector embeddings
- Hybrid retrieval (BM25 + dense retrieval)
- Maximum Marginal Relevance (MMR)
- Cross-encoder reranking
- Retrieval-Augmented Generation (RAG)
- Tool-calling AI agent
- Web search and calculator tools
- Statistical analysis tool
- LangGraph orchestration
- Human-in-the-loop approval

## Tech Stack

- Python
- LangChain
- LangGraph
- Groq
- ChromaDB
- Hugging Face Sentence Transformers
- Streamlit

## Installation

Clone the repository:

    git clone YOUR_REPOSITORY_URL
    cd ResearchMind

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements-lock.txt

Configure environment variables:

    Copy-Item .env.example .env

Add your Groq credentials to .env.

Run the application:

    streamlit run ui/app.py

## Project Status

Core RAG, retrieval, tool calling and initial LangGraph
human-in-the-loop integration are implemented.

Deployment and advanced agent capabilities are in progress.
