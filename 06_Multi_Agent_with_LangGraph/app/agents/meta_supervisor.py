"""Meta-supervisor for coordinating content and verification teams."""

import functools
from typing import Dict, Any

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from app.models.state import LinkedInMetaState
from app.agents.helpers import create_team_supervisor, get_llm, get_last_message, join_graph
from app.agents.content_team import content_chain
from app.agents.verification_team import verification_chain


def create_linkedin_meta_graph():
    """
    Create and compile the meta-supervisor graph for LinkedIn post generation.
    
    Returns:
        Compiled LangGraph for the complete LinkedIn post generation system
    """
    llm = get_llm()
    
    # Create meta-supervisor to coordinate content and verification teams
    linkedin_meta_supervisor = create_team_supervisor(
        llm,
        ("You are a meta-supervisor managing LinkedIn post generation. You coordinate between "
         "the following teams: {team_members}. First direct the Content team to research a paper "
         "and create a LinkedIn post. Then send the completed post to the Verification team to check "
         "technical accuracy and LinkedIn style compliance. "
         "The workflow should be: Content team → Verification team → FINISH. "
         "Only finish when both teams have completed their work successfully."),
        ["Content team", "Verification team"],
    )
    
    # Create the meta-graph
    linkedin_meta_graph = StateGraph(LinkedInMetaState)
    
    # Add team nodes to meta-graph (these are entire sub-graphs)
    linkedin_meta_graph.add_node(
        "Content team", 
        get_last_message | content_chain | join_graph
    )
    linkedin_meta_graph.add_node(
        "Verification team", 
        get_last_message | verification_chain | join_graph
    )
    linkedin_meta_graph.add_node("meta_supervisor", linkedin_meta_supervisor)
    
    # Add edges for the meta-workflow
    linkedin_meta_graph.add_edge("Content team", "meta_supervisor")
    linkedin_meta_graph.add_edge("Verification team", "meta_supervisor")
    linkedin_meta_graph.add_conditional_edges(
        "meta_supervisor",
        lambda x: x["next"],
        {
            "Content team": "Content team",
            "Verification team": "Verification team",
            "FINISH": END,
        },
    )
    
    # Set entry point and compile
    linkedin_meta_graph.set_entry_point("meta_supervisor")
    return linkedin_meta_graph.compile()


def create_linkedin_post_request(
    paper_title: str,
    additional_context: str = None,
    target_audience: str = "professional",
    include_technical_details: bool = True,
    max_hashtags: int = 10,
    tone: str = "professional"
) -> str:
    """
    Create a formatted request message for LinkedIn post generation.
    
    Args:
        paper_title: Title of the ML paper
        additional_context: Additional context or focus areas
        target_audience: Target audience for the post
        include_technical_details: Whether to include technical details
        max_hashtags: Maximum number of hashtags
        tone: Tone of the post
        
    Returns:
        Formatted request message
    """
    request_parts = [
        f"Create a LinkedIn post about the machine learning paper: '{paper_title}'"
    ]
    
    if additional_context:
        request_parts.append(f"Additional context: {additional_context}")
    
    request_parts.extend([
        f"Target audience: {target_audience}",
        f"Include technical details: {'Yes' if include_technical_details else 'No'}",
        f"Maximum hashtags: {max_hashtags}",
        f"Tone: {tone}",
        "",
        "Process:",
        "1. First, research the paper thoroughly to understand its methodology, results, and impact",
        "2. Create an engaging LinkedIn post based on the research",
        "3. Verify the technical accuracy of all claims",
        "4. Check that the post follows LinkedIn style best practices",
        "5. Provide the final, verified post ready for publication"
    ])
    
    return "\n".join(request_parts)


def enter_linkedin_meta_state(message: str, task_id: str = None) -> Dict[str, Any]:
    """
    Create entry point for the LinkedIn meta-graph.
    
    Args:
        message: Input message
        task_id: Optional task ID for tracking
        
    Returns:
        Dictionary formatted for LinkedInMetaState
    """
    return {
        "messages": [HumanMessage(content=message)],
        "next": "",
        "task_id": task_id,
        "paper_title": None,
        "current_step": "initializing",
        "progress": 0.0,
        "research_complete": False,
        "content_complete": False,
        "verification_complete": False,
        "final_post": None,
        "error_message": None,
    }


# Create the compiled LinkedIn post generation graph
compiled_linkedin_graph = create_linkedin_meta_graph()


def run_linkedin_post_generation(
    paper_title: str,
    additional_context: str = None,
    target_audience: str = "professional",
    include_technical_details: bool = True,
    max_hashtags: int = 10,
    tone: str = "professional",
    task_id: str = None
) -> Dict[str, Any]:
    """
    Run the complete LinkedIn post generation process.
    
    Args:
        paper_title: Title of the ML paper
        additional_context: Additional context or focus areas
        target_audience: Target audience for the post
        include_technical_details: Whether to include technical details
        max_hashtags: Maximum number of hashtags
        tone: Tone of the post
        task_id: Optional task ID for tracking
        
    Returns:
        Complete results from the LinkedIn post generation process
    """
    # Create the request message
    request_message = create_linkedin_post_request(
        paper_title=paper_title,
        additional_context=additional_context,
        target_audience=target_audience,
        include_technical_details=include_technical_details,
        max_hashtags=max_hashtags,
        tone=tone
    )
    
    # Create initial state
    initial_state = enter_linkedin_meta_state(request_message, task_id)
    
    # Run the complete workflow
    try:
        result = compiled_linkedin_graph.invoke(
            initial_state,
            {"recursion_limit": 50}
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "task_id": task_id,
            "status": "failed"
        } 