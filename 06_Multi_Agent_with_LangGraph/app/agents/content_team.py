"""Content creation team for LinkedIn ML Paper Post Generation."""

import functools
from typing import List

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from app.models.state import ContentTeamState
from app.tools.search_tools import research_ml_paper, tavily_tool
from app.tools.linkedin_tools import create_linkedin_post
from app.agents.helpers import create_agent, create_team_supervisor, get_llm, create_partial_agent_node


def create_content_team_graph():
    """
    Create and compile the content creation team graph.
    
    Returns:
        Compiled LangGraph for content creation
    """
    llm = get_llm()
    
    # Create Paper Researcher Agent
    paper_researcher_agent = create_agent(
        llm,
        [research_ml_paper, tavily_tool],
        ("You are an expert AI researcher who specializes in understanding and summarizing "
         "machine learning papers. Your job is to research papers thoroughly and extract "
         "key insights, methodologies, and results. Focus on accuracy and clarity. "
         "Always provide comprehensive information about the paper including its main "
         "contributions, methodology, results, and potential impact.")
    )
    paper_researcher_node = create_partial_agent_node(paper_researcher_agent, "PaperResearcher")
    
    # Create LinkedIn Post Creator Agent  
    linkedin_creator_agent = create_agent(
        llm,
        [create_linkedin_post],
        ("You are a social media expert who specializes in creating engaging LinkedIn posts "
         "about technical topics. You know how to make complex AI research accessible and "
         "engaging for a professional audience. Create posts that drive engagement while "
         "maintaining technical accuracy. Always include relevant hashtags and ask "
         "engaging questions to encourage comments and discussions.")
    )
    linkedin_creator_node = create_partial_agent_node(linkedin_creator_agent, "LinkedInCreator")
    
    # Create Content Team Supervisor
    content_supervisor = create_team_supervisor(
        llm,
        ("You are a supervisor managing a content creation team with the following workers: {team_members}. "
         "Your job is to coordinate research and post creation. First have the researcher gather information "
         "about the paper, then have the creator make a LinkedIn post based on that research. "
         "Ensure the research is thorough before moving to content creation. "
         "When both research and post creation are complete, respond with FINISH."),
        ["PaperResearcher", "LinkedInCreator"],
    )
    
    # Build the content creation graph
    content_graph = StateGraph(ContentTeamState)
    
    # Add nodes
    content_graph.add_node("PaperResearcher", paper_researcher_node)
    content_graph.add_node("LinkedInCreator", linkedin_creator_node)
    content_graph.add_node("content_supervisor", content_supervisor)
    
    # Add edges for content creation flow
    content_graph.add_edge("PaperResearcher", "content_supervisor")
    content_graph.add_edge("LinkedInCreator", "content_supervisor")
    content_graph.add_conditional_edges(
        "content_supervisor",
        lambda x: x["next"],
        {
            "PaperResearcher": "PaperResearcher",
            "LinkedInCreator": "LinkedInCreator", 
            "FINISH": END,
        },
    )
    
    # Set entry point and compile
    content_graph.set_entry_point("content_supervisor")
    return content_graph.compile()


def enter_content_team(message: str, members: List[str]):
    """
    Create entry point for the content team graph.
    
    Args:
        message: Input message for the team
        members: List of team member names
        
    Returns:
        Dictionary formatted for the content team state
    """
    return {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
        "next": "",
        "paper_title": None,
        "research_findings": None,
        "draft_post": None,
    }


# Create the compiled content team graph
compiled_content_graph = create_content_team_graph()

# Create the content team chain interface
content_chain = (
    functools.partial(enter_content_team, members=["PaperResearcher", "LinkedInCreator"])
    | compiled_content_graph
) 