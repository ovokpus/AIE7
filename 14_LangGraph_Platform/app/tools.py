"""Toolbelt assembly for agents.

Collects third-party tools, local tools (like RAG), and MCP tools into a single list that
graphs can bind to their language models.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag import retrieve_information
from app.mcp_client import get_mcp_tools

logger = logging.getLogger(__name__)


def get_tool_belt(include_mcp: bool = True) -> List:
    """
    Return the list of tools available to agents.
    
    Includes:
    - Tavily Search: Web search capabilities
    - Arxiv: Academic paper search
    - RAG: Information retrieval from local documents
    - MCP Tools: File operations, data analysis, system utilities (optional)
    
    Args:
        include_mcp: Whether to include MCP tools (default: True)
        
    Returns:
        List of tools for the agent
    """
    # Core tools
    tavily_tool = TavilySearchResults(max_results=5)
    core_tools = [tavily_tool, ArxivQueryRun(), retrieve_information]
    
    # Add MCP tools if requested
    if include_mcp:
        try:
            mcp_tools = get_mcp_tools()
            logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools")
            core_tools.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}")
            logger.info("Continuing with core tools only")
    
    return core_tools


def get_core_tool_belt() -> List:
    """Return only the core tools (Tavily, Arxiv, RAG) without MCP tools."""
    return get_tool_belt(include_mcp=False)


