"""Verification team for LinkedIn ML Paper Post Generation."""

import functools
from typing import List

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from app.models.state import VerificationTeamState
from app.tools.linkedin_tools import verify_technical_accuracy, check_linkedin_style
from app.tools.search_tools import research_ml_paper
from app.agents.helpers import create_agent, create_team_supervisor, get_llm, create_partial_agent_node


def create_verification_team_graph():
    """
    Create and compile the verification team graph.
    
    Returns:
        Compiled LangGraph for verification
    """
    llm = get_llm()
    
    # Create Technical Accuracy Checker Agent
    tech_verifier_agent = create_agent(
        llm,
        [verify_technical_accuracy, research_ml_paper],
        ("You are a technical reviewer and fact-checker specializing in machine learning research. "
         "Your job is to verify that LinkedIn posts accurately represent the research they discuss. "
         "Check for technical accuracy, proper methodology description, and correct representation "
         "of results. Flag any oversimplified or incorrect claims. Ensure proper attribution to "
         "authors and avoid overstated language. Be thorough in your analysis and provide specific "
         "recommendations for improvement.")
    )
    tech_verifier_node = create_partial_agent_node(tech_verifier_agent, "TechVerifier")
    
    # Create LinkedIn Style Checker Agent
    style_checker_agent = create_agent(
        llm,
        [check_linkedin_style],
        ("You are a LinkedIn content strategist who ensures posts follow best practices for "
         "professional social media. Check for appropriate tone, formatting, hashtag usage, "
         "engagement elements, and overall LinkedIn style compliance. Suggest improvements "
         "to maximize professional impact and engagement. Focus on readability, professional "
         "presentation, and LinkedIn-specific optimization techniques.")
    )
    style_checker_node = create_partial_agent_node(style_checker_agent, "StyleChecker")
    
    # Create Verification Team Supervisor
    verification_supervisor = create_team_supervisor(
        llm,
        ("You are a supervisor managing a verification team with the following workers: {team_members}. "
         "Your job is to ensure quality control for LinkedIn posts about ML research. "
         "Have the technical verifier check accuracy first, then have the style checker ensure "
         "LinkedIn compliance. Both verifications must be completed before finishing. "
         "When both technical and style verifications are complete, respond with FINISH."),
        ["TechVerifier", "StyleChecker"],
    )
    
    # Build the verification graph
    verification_graph = StateGraph(VerificationTeamState)
    
    # Add nodes
    verification_graph.add_node("TechVerifier", tech_verifier_node)
    verification_graph.add_node("StyleChecker", style_checker_node)
    verification_graph.add_node("verification_supervisor", verification_supervisor)
    
    # Add edges for verification flow
    verification_graph.add_edge("TechVerifier", "verification_supervisor")
    verification_graph.add_edge("StyleChecker", "verification_supervisor")
    verification_graph.add_conditional_edges(
        "verification_supervisor",
        lambda x: x["next"],
        {
            "TechVerifier": "TechVerifier",
            "StyleChecker": "StyleChecker",
            "FINISH": END,
        },
    )
    
    # Set entry point and compile
    verification_graph.set_entry_point("verification_supervisor")
    return verification_graph.compile()


def enter_verification_team(message: str, members: List[str]):
    """
    Create entry point for the verification team graph.
    
    Args:
        message: Input message for the team
        members: List of team member names
        
    Returns:
        Dictionary formatted for the verification team state
    """
    return {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
        "next": "",
        "post_content": None,
        "technical_report": None,
        "style_report": None,
        "verification_complete": False,
    }


# Create the compiled verification team graph
compiled_verification_graph = create_verification_team_graph()

# Create the verification team chain interface
verification_chain = (
    functools.partial(enter_verification_team, members=["TechVerifier", "StyleChecker"])
    | compiled_verification_graph
) 