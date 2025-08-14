"""Agent Tool Belt Assembly and Management.

This module provides a centralized tool collection for the LangGraph agent,
combining third-party tools (Tavily web search, ArXiv academic search) with
custom tools (RAG document retrieval) into a unified interface.

The tool belt enables the agent to:
    - Search the web for current information via Tavily API
    - Query academic papers from ArXiv repository
    - Retrieve relevant information from local document store via RAG

Tools are designed to work seamlessly within the LangGraph workflow and
support the A2A protocol's capability discovery and execution patterns.

Available Tools:
    - TavilySearchResults: Web search with configurable result limits
    - ArxivQueryRun: Academic paper search and retrieval
    - retrieve_information: RAG-based document search and synthesis

Example:
    >>> tools = get_tool_belt()
    >>> model_with_tools = model.bind_tools(tools)
    >>> # Agent can now access web search, ArXiv, and RAG capabilities
"""
from __future__ import annotations

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag import retrieve_information


def get_tool_belt() -> List:
    """Assemble and return the complete tool collection for LangGraph agents.
    
    Creates a comprehensive tool belt containing web search, academic search,
    and document retrieval capabilities. Each tool is pre-configured with
    optimal settings for the agent's use cases.
    
    Returns:
        List: Collection of LangChain tools including:
            - TavilySearchResults: Web search limited to 5 results for efficiency
            - ArxivQueryRun: Academic paper search with default configuration
            - retrieve_information: Custom RAG tool for document retrieval
            
    Note:
        Tools require appropriate API keys:
        - TAVILY_API_KEY for web search functionality
        - OPENAI_API_KEY for RAG embeddings and LLM operations
        ArXiv tool requires no API key as it uses public access.
        
    Example:
        >>> tools = get_tool_belt()
        >>> len(tools)
        3
        >>> model_with_tools = chat_model.bind_tools(tools)
        >>> # Agent can now call web search, ArXiv, and RAG
    """
    tavily_tool = TavilySearchResults(max_results=5)
    return [tavily_tool, ArxivQueryRun(), retrieve_information]
