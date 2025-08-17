"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import evaluate, Client
from langsmith.schemas import Run, Example
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    max_loops: int = 3,
    helpfulness_threshold: float = 7.0
):
    """Create a LangGraph agent with helpfulness evaluation loop.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        max_loops: Maximum number of helpfulness evaluation loops
        
    Returns:
        Compiled LangGraph agent with helpfulness checking
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"
    
    def helpfulness_evaluator(inputs: dict, outputs: dict) -> dict:
        """LangSmith evaluator for helpfulness assessment."""
        try:
            prompt = f"""You are an expert evaluator assessing the helpfulness of AI responses.

User's question: {inputs.get('question', '')}
AI response: {outputs.get('response', '')}

Evaluate the response based on these criteria:
1. Relevance: Does it directly address the user's question?
2. Completeness: Is the information comprehensive and complete?
3. Clarity: Is it clear and easy to understand?
4. Actionability: Does it provide useful, actionable guidance when appropriate?
5. Accuracy: Is the information correct and well-sourced?

Rate the overall helpfulness on a scale of 1-10 (1=not helpful at all, 10=extremely helpful).

Provide ONLY your numerical rating as: SCORE: X"""
            
            eval_model = get_openai_model(model_name="gpt-4.1-mini", temperature=0.0)
            response = eval_model.invoke(prompt)
            
            content = response.content.upper()
            score = 5.0
            if "SCORE:" in content:
                try:
                    score_part = content.split("SCORE:")[1].strip()
                    if "/" in score_part:
                        score_part = score_part.split("/")[0]
                    score = float(score_part)
                    score = max(1.0, min(10.0, score))
                except:
                    score = 5.0
            
            return {
                "helpfulness_score": score,
                "evaluation_reasoning": response.content
            }
        except Exception as e:
            # Default to helpful to avoid infinite loops
            return {
                "helpfulness_score": 7.0,
                "evaluation_reasoning": f"Evaluation failed: {str(e)}, defaulting to helpful"
            }
    
    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness using LangSmith evaluator."""
        # Count how many helpfulness evaluations we've done
        helpfulness_count = sum(1 for m in state["messages"] 
                              if hasattr(m, 'content') and 'HELPFULNESS:' in str(m.content))
        
        # If we've exceeded loop limit, short-circuit with END decision marker
        if helpfulness_count >= max_loops:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}
        
        # Find the initial human query
        initial_query = None
        for msg in state["messages"]:
            if hasattr(msg, 'content') and not str(msg.content).startswith('HELPFULNESS:'):
                if not getattr(msg, 'tool_calls', None):  # Skip tool calls
                    initial_query = msg
                    break
        
        if not initial_query:
            return {"messages": [AIMessage(content="HELPFULNESS:Y")]}  # No query found, end
        
        # Get the latest non-helpfulness response
        final_response = None
        for msg in reversed(state["messages"]):
            if (hasattr(msg, 'content') and 
                not str(msg.content).startswith('HELPFULNESS:') and
                not getattr(msg, 'tool_calls', None)):
                final_response = msg
                break
        
        if not final_response:
            return {"messages": [AIMessage(content="HELPFULNESS:Y")]}  # No response found, end
        
        try:
            # Use external helpfulness evaluator for consistency with notebook
            evaluation_result = helpfulness_evaluator(
                inputs={"question": initial_query.content},
                outputs={"response": final_response.content}
            )
            
            # Use raw score instead of boolean
            score = evaluation_result.get("helpfulness_score", 7.0)
            reasoning = evaluation_result.get("evaluation_reasoning", "")
            is_helpful = score >= helpfulness_threshold
            
            # Log evaluation to LangSmith if client is available
            try:
                client = Client()
                client.create_run(
                    name="helpfulness_evaluation",
                    inputs={"question": initial_query.content, "response": final_response.content},
                    outputs={"score": score, "is_helpful": is_helpful, "reasoning": reasoning},
                    run_type="llm"
                )
            except Exception:
                # LangSmith logging is optional - continue if it fails
                pass
            
            if is_helpful:
                return {"messages": [AIMessage(content=f"HELPFULNESS:Y:SCORE:{score}")]}
            else:
                # Add refinement instruction for low scores
                refinement_instruction = f"""The previous response scored {score}/10 for helpfulness. Please provide a more comprehensive, clearer, and more actionable response to the original question: "{initial_query.content}"

Previous response: {final_response.content}

Improve the response by making it more relevant, complete, clear, and actionable."""
                
                return {"messages": [
                    AIMessage(content=f"HELPFULNESS:N:SCORE:{score}"),
                    HumanMessage(content=refinement_instruction)
                ]}
            
        except Exception as e:
            # If evaluation fails, default to helpful to avoid infinite loops
            return {"messages": [AIMessage(content="HELPFULNESS:Y:SCORE:7.0")]}
    
    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check for loop-limit marker
        last_messages = state["messages"][-1:]
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in last_messages):
            return END
        
        last = state["messages"][-1]
        content = getattr(last, "content", "")
        if "HELPFULNESS:Y" in content:
            return END
        elif "HELPFULNESS:N" in content:
            return "agent"  # Continue to improve response
        return END  # Default to end if unclear
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    
    graph.set_entry_point("agent")
    
    # Route from agent to either tools or helpfulness check
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"}
    )
    
    # Route from helpfulness check to either continue or end
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"agent": "agent", END: END}
    )
    
    # After tool execution, go back to agent
    graph.add_edge("action", "agent")
    
    return graph.compile()
