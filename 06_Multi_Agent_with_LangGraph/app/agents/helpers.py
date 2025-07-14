"""Helper functions for creating and managing agents in the multi-agent system."""

import functools
from typing import Any, Callable, List, Optional, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from app.config import settings


def agent_node(state, agent, name):
    """
    Create a node that wraps an agent executor.
    
    Args:
        state: The current state
        agent: The agent executor to run
        name: Name of the agent for the message
        
    Returns:
        Dictionary with messages containing the agent's output
    """
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> AgentExecutor:
    """
    Create a function-calling agent and return its executor.
    
    Args:
        llm: The language model to use
        tools: List of tools available to the agent
        system_prompt: System prompt for the agent
        
    Returns:
        AgentExecutor configured with the specified tools and prompt
    """
    # Add standard suffix for autonomous operation
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason!"
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    return executor


def create_team_supervisor(llm: ChatOpenAI, system_prompt: str, members: List[str]) -> Runnable:
    """
    Create an LLM-based router/supervisor for a team of agents.
    
    Args:
        llm: The language model to use for decisions
        system_prompt: Instructions for the supervisor
        members: List of team member names to route between
        
    Returns:
        Runnable that can route to team members or finish
    """
    options = ["FINISH"] + members
    
    # Define the function for routing decisions
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), team_members=", ".join(members))
    
    return (
        prompt
        | llm.bind_tools([function_def], tool_choice="route")
        | JsonOutputFunctionsParser()
    )


def get_llm() -> ChatOpenAI:
    """
    Get configured ChatOpenAI instance.
    
    Returns:
        Configured ChatOpenAI model
    """
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature
    )


def get_last_message(state: dict) -> str:
    """
    Extract the content of the last message from state.
    
    Args:
        state: State dictionary containing messages
        
    Returns:
        Content of the last message
    """
    return state["messages"][-1].content


def join_graph(response: dict) -> dict:
    """
    Extract the last message from a subgraph response.
    
    Args:
        response: Response from a subgraph
        
    Returns:
        Dictionary with the last message
    """
    return {"messages": [response["messages"][-1]]}


def enter_chain(message: str, members: Optional[List[str]] = None) -> dict:
    """
    Create entry point for a chain/graph.
    
    Args:
        message: Input message
        members: List of team members (optional)
        
    Returns:
        Dictionary formatted for graph entry
    """
    results = {
        "messages": [HumanMessage(content=message)],
    }
    
    if members:
        results["team_members"] = ", ".join(members)
    
    return results


def create_partial_agent_node(agent: AgentExecutor, name: str) -> Callable:
    """
    Create a partial function for an agent node.
    
    Args:
        agent: The agent executor
        name: Name of the agent
        
    Returns:
        Partial function ready to be used as a node
    """
    return functools.partial(agent_node, agent=agent, name=name) 