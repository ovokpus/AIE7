from typing import Annotated, List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from app.config import settings
import os


# Initialize Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)


@tool
def research_ml_paper(
    paper_title: Annotated[str, "Title or topic of the ML paper to research"],
    focus_areas: Annotated[Optional[List[str]], "Specific aspects to focus on"] = None,
    search_depth: Annotated[str, "Search depth: basic, detailed, comprehensive"] = "detailed"
) -> Annotated[str, "Research findings about the ML paper"]:
    """Research a machine learning paper using web search to gather comprehensive information."""
    
    if focus_areas is None:
        focus_areas = ["methodology", "results", "applications", "impact"]
    
    # Build search query
    if search_depth == "basic":
        search_query = f"machine learning paper {paper_title} summary"
    elif search_depth == "comprehensive":
        search_query = f"machine learning paper {paper_title} arxiv research methodology results applications impact"
    else:  # detailed
        search_query = f"machine learning paper {paper_title} arxiv research"
    
    try:
        # Use Tavily to search for information
        search_results = tavily_tool.invoke(search_query)
        
        # Enhanced search for specific focus areas
        enhanced_results = []
        for area in focus_areas:
            area_query = f"{paper_title} {area} machine learning"
            try:
                area_results = tavily_tool.invoke(area_query)
                enhanced_results.append(f"\n--- {area.upper()} ---\n{area_results}")
            except Exception as e:
                enhanced_results.append(f"\n--- {area.upper()} ---\nError searching for {area}: {str(e)}")
        
        # Combine all results
        combined_results = f"""
MAIN RESEARCH FINDINGS:
{search_results}

FOCUSED RESEARCH AREAS:
{chr(10).join(enhanced_results)}
"""
        
        return combined_results
        
    except Exception as e:
        return f"Error researching paper '{paper_title}': {str(e)}"


@tool
def search_arxiv_papers(
    query: Annotated[str, "Search query for ArXiv papers"],
    max_results: Annotated[int, "Maximum number of results to return"] = 5
) -> Annotated[str, "ArXiv search results"]:
    """Search for papers on ArXiv using the provided query."""
    
    try:
        # Search for ArXiv papers specifically
        arxiv_query = f"site:arxiv.org {query} machine learning"
        search_results = tavily_tool.invoke(arxiv_query)
        
        return f"ArXiv Search Results for '{query}':\n{search_results}"
        
    except Exception as e:
        return f"Error searching ArXiv for '{query}': {str(e)}"


@tool
def get_paper_citations(
    paper_title: Annotated[str, "Title of the paper to get citations for"]
) -> Annotated[str, "Citation information and impact metrics"]:
    """Get citation information and impact metrics for a research paper."""
    
    try:
        # Search for citation information
        citation_query = f"{paper_title} citations impact factor google scholar"
        search_results = tavily_tool.invoke(citation_query)
        
        return f"Citation Information for '{paper_title}':\n{search_results}"
        
    except Exception as e:
        return f"Error getting citations for '{paper_title}': {str(e)}"


@tool  
def search_recent_developments(
    topic: Annotated[str, "ML topic to search for recent developments"],
    time_period: Annotated[str, "Time period: recent, 2024, 2023"] = "recent"
) -> Annotated[str, "Recent developments in the specified topic"]:
    """Search for recent developments and trends in a specific ML topic."""
    
    try:
        # Build time-specific query
        if time_period == "2024":
            time_filter = "2024"
        elif time_period == "2023":  
            time_filter = "2023"
        else:
            time_filter = "recent latest new"
        
        developments_query = f"{topic} machine learning {time_filter} developments trends"
        search_results = tavily_tool.invoke(developments_query)
        
        return f"Recent Developments in '{topic}':\n{search_results}"
        
    except Exception as e:
        return f"Error searching recent developments for '{topic}': {str(e)}"


@tool
def search_paper_applications(
    paper_title: Annotated[str, "Title of the paper"],
    industry: Annotated[Optional[str], "Specific industry to focus on"] = None
) -> Annotated[str, "Real-world applications and use cases"]:
    """Search for real-world applications and use cases of a research paper."""
    
    try:
        if industry:
            app_query = f"{paper_title} applications use cases {industry} implementation"
        else:
            app_query = f"{paper_title} applications use cases real world implementation"
        
        search_results = tavily_tool.invoke(app_query)
        
        return f"Applications and Use Cases for '{paper_title}':\n{search_results}"
        
    except Exception as e:
        return f"Error searching applications for '{paper_title}': {str(e)}"


def validate_search_tools():
    """Validate that search tools are properly configured."""
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is required for search functionality")
    
    # Test basic search functionality
    try:
        test_result = tavily_tool.invoke("machine learning test")
        return True
    except Exception as e:
        raise RuntimeError(f"Search tools validation failed: {e}")


# Export commonly used tools
__all__ = [
    'research_ml_paper',
    'search_arxiv_papers', 
    'get_paper_citations',
    'search_recent_developments',
    'search_paper_applications',
    'tavily_tool',
    'validate_search_tools'
] 