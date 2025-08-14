"""Retrieval-Augmented Generation (RAG) Implementation.

This module provides a complete RAG (Retrieval-Augmented Generation) pipeline
for document-based question answering. It combines document ingestion, vector
storage, similarity search, and context-aware response generation.

The RAG system architecture:
    1. Document Loading: PDF files loaded from configurable directory
    2. Text Chunking: Token-aware splitting for optimal retrieval
    3. Vector Embedding: OpenAI embeddings stored in Qdrant vector database
    4. Retrieval: Similarity search for relevant document chunks
    5. Generation: Context-constrained LLM responses

Key Components:
    - PDF document loader with automatic text extraction
    - Token-aware text chunking using tiktoken
    - In-memory Qdrant vector store for fast retrieval
    - LangChain tool integration for agent workflows
    - Context-aware response generation with source attribution

Environment Configuration:
    - RAG_DATA_DIR: Directory containing PDF documents (default: "data")
    - OPENAI_API_KEY: Required for embeddings and LLM generation
    - Optional: Qdrant connection settings for persistent storage

Usage:
    The module exposes `retrieve_information` as a LangChain tool that can be
    called by agents to search documents and generate contextual responses.

Example:
    >>> # Tool is automatically loaded in agent workflow
    >>> response = retrieve_information("What are student loan requirements?")
    >>> # Returns context-aware answer based on loaded PDF documents
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, List

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken; used for chunk length measurement."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: List[Document]
    response: str


def _build_rag_graph(data_dir: str):
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    # Split documents
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # Fallback to legacy import path if available
        from langchain.text_splitter import (  # type: ignore
            RecursiveCharacterTextSplitter,
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    # Embeddings and vector store (in-memory Qdrant)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:"
    )
    retriever = qdrant_vectorstore.as_retriever()

    # Prompt and model
    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-nano"))

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about student loan policies"""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result
