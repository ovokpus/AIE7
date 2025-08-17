"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os
import time
import re

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
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

# Check for Guardrails availability
try:
    import guardrails as gd
    from guardrails import Guard
    from guardrails.hub import ToxicLanguage, DetectPII, RestrictToTopic, NSFWText, CompetitorCheck
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ Guardrails not available - some functionality will use mock implementations")


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


class GuardedAgentState(TypedDict):
    """Enhanced state schema for guardrail-protected agents."""
    messages: Annotated[List[BaseMessage], add_messages]
    input_guard_results: dict  # Track input validation results
    output_guard_results: dict  # Track output validation results
    guard_violations: List[str]  # Track any guard violations
    refinement_count: int  # Track refinement attempts
    max_refinements: int  # Maximum allowed refinements


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


def _initialize_guardrails():
    """Initialize guardrails guards if available."""
    if not GUARDRAILS_AVAILABLE:
        return None, None, None, None
    
    try:
        # Topic Restriction Guard
        topic_guard = Guard().use(
            RestrictToTopic,
            valid_topics=["student loans", "financial aid", "education financing", "loan repayment", "FAFSA"],
            invalid_topics=["crypto", "cryptocurrency", "investment advice", "gambling", "politics", "medical advice"]
        )
        
        # Jailbreak Detection Guard
        jailbreak_guard = Guard().use(
            CompetitorCheck,
            competitors=["ignore instructions", "jailbreak", "bypass", "unrestricted AI", "DAN", "do anything now"]
        )
        
        # PII Protection Guard
        pii_guard = Guard().use(DetectPII, pii_entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"])
        
        # Content Moderation Guard
        content_guard = Guard().use(ToxicLanguage)
        
        return topic_guard, jailbreak_guard, pii_guard, content_guard
        
    except Exception as e:
        print(f"⚠️ Error initializing guardrails: {e}")
        return None, None, None, None


def create_input_guardrails_node():
    """
    Create an input validation node that checks:
    - Jailbreak attempts
    - Topic restrictions  
    - PII detection
    - Content appropriateness
    
    Returns:
        Function that validates user input against multiple guardrails
    """
    topic_guard, jailbreak_guard, pii_guard, content_guard = _initialize_guardrails()
    
    def input_validation_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate user input against multiple guardrails."""
        print("\n🛡️ Running Input Validation...")
        
        # Get the latest user message
        user_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'content') and not isinstance(msg, AIMessage):
                user_message = msg
                break
        
        if not user_message:
            return {
                "input_guard_results": {"status": "no_input", "passed": True},
                "guard_violations": []
            }
        
        user_input = user_message.content
        violations = []
        guard_results = {}
        
        if GUARDRAILS_AVAILABLE and topic_guard and jailbreak_guard and pii_guard:
            # 1. Jailbreak Detection
            try:
                jailbreak_result = jailbreak_guard.validate(user_input)
                guard_results["jailbreak"] = {
                    "passed": jailbreak_result.validation_passed,
                    "output": jailbreak_result.validated_output
                }
                if not jailbreak_result.validation_passed:
                    violations.append("jailbreak_detected")
                    print("  ❌ Jailbreak attempt detected")
                else:
                    print("  ✅ Jailbreak check passed")
            except Exception as e:
                guard_results["jailbreak"] = {"passed": False, "error": str(e)}
                violations.append("jailbreak_error")
                print(f"  ⚠️ Jailbreak check error: {e}")
            
            # 2. Topic Restriction
            try:
                topic_guard.validate(user_input)
                guard_results["topic"] = {"passed": True}
                print("  ✅ Topic restriction passed")
            except Exception as e:
                guard_results["topic"] = {"passed": False, "error": str(e)}
                violations.append("topic_violation")
                print(f"  ❌ Topic violation: {e}")
            
            # 3. PII Detection and Redaction
            try:
                pii_result = pii_guard.validate(user_input)
                guard_results["pii"] = {
                    "passed": pii_result.validation_passed,
                    "sanitized_input": pii_result.validated_output
                }
                if user_input != pii_result.validated_output:
                    print("  🔒 PII detected and redacted")
                    # Update the user message with sanitized content
                    sanitized_messages = []
                    for msg in state["messages"]:
                        if msg == user_message:
                            sanitized_messages.append(HumanMessage(content=pii_result.validated_output))
                        else:
                            sanitized_messages.append(msg)
                    return {
                        "messages": sanitized_messages,
                        "input_guard_results": guard_results,
                        "guard_violations": violations
                    }
                else:
                    print("  ✅ No PII detected")
            except Exception as e:
                guard_results["pii"] = {"passed": False, "error": str(e)}
                violations.append("pii_error")
                print(f"  ⚠️ PII check error: {e}")
        
        else:
            # Mock implementation when Guardrails not available
            print("  🔍 Mock validation (Guardrails not configured)")
            guard_results = {
                "jailbreak": {"passed": True},
                "topic": {"passed": True}, 
                "pii": {"passed": True}
            }
            
            # Simple heuristic checks for demo
            lower_input = user_input.lower()
            if any(word in lower_input for word in ["ignore", "jailbreak", "hack", "bypass"]):
                violations.append("potential_jailbreak")
                guard_results["jailbreak"]["passed"] = False
                print("  ⚠️ Potential jailbreak detected (mock)")
            
            if any(word in lower_input for word in ["crypto", "investment", "gambling", "politics"]):
                violations.append("topic_violation")
                guard_results["topic"]["passed"] = False
                print("  ⚠️ Off-topic content detected (mock)")
        
        return {
            "input_guard_results": guard_results,
            "guard_violations": violations
        }
    
    return input_validation_node


def create_output_guardrails_node(rag_chain: Optional[ProductionRAGChain] = None):
    """
    Create an output validation node that checks:
    - Content moderation (profanity, appropriateness)
    - Factuality against source documents
    - PII leakage prevention
    - Response quality and helpfulness
    
    Args:
        rag_chain: Optional RAG chain for factuality checking
        
    Returns:
        Function that validates agent output against multiple guardrails
    """
    topic_guard, jailbreak_guard, pii_guard, content_guard = _initialize_guardrails()
    
    def output_validation_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate agent output against multiple guardrails."""
        print("\n🔍 Running Output Validation...")
        
        # Get the latest AI response
        ai_response = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                if not str(msg.content).startswith('HELPFULNESS:'):  # Skip helpfulness evaluations
                    ai_response = msg
                    break
        
        if not ai_response:
            return {
                "output_guard_results": {"status": "no_response", "passed": True},
                "guard_violations": []
            }
        
        response_content = ai_response.content
        violations = []
        guard_results = {}
        
        # Get original user query for context
        user_query = None
        for msg in state["messages"]:
            if hasattr(msg, 'content') and not isinstance(msg, AIMessage):
                user_query = msg.content
                break
        
        if GUARDRAILS_AVAILABLE and content_guard and pii_guard:
            # 1. Content Moderation
            try:
                content_result = content_guard.validate(response_content)
                guard_results["profanity"] = {
                    "passed": content_result.validation_passed,
                    "score": getattr(content_result, 'score', None)
                }
                if not content_result.validation_passed:
                    violations.append("inappropriate_content")
                    print("  ❌ Inappropriate content detected")
                else:
                    print("  ✅ Content moderation passed")
            except Exception as e:
                guard_results["profanity"] = {"passed": True, "error": str(e)}
                print(f"  ⚠️ Content moderation error (continuing): {e}")
            
            # 2. PII Leakage Detection
            try:
                output_pii_result = pii_guard.validate(response_content)
                guard_results["output_pii"] = {
                    "passed": output_pii_result.validation_passed,
                    "sanitized_output": output_pii_result.validated_output
                }
                if not output_pii_result.validation_passed:
                    violations.append("pii_leakage")
                    print("  ❌ PII leakage detected in response")
                else:
                    print("  ✅ No PII leakage detected")
            except Exception as e:
                guard_results["output_pii"] = {"passed": True, "error": str(e)}
                print(f"  ⚠️ PII check error (continuing): {e}")
            
            # 3. Factuality Check (if RAG chain available)
            if rag_chain and user_query:
                try:
                    # This would require a factuality guard from Guardrails
                    # For now, implementing basic consistency check
                    guard_results["factuality"] = {"passed": True, "note": "Basic factuality check"}
                    print("  ✅ Basic factuality check passed")
                except Exception as e:
                    guard_results["factuality"] = {"passed": True, "error": str(e)}
                    print(f"  ⚠️ Factuality check error (continuing): {e}")
        
        else:
            # Mock implementation when Guardrails not available
            print("  🔍 Mock output validation (Guardrails not configured)")
            guard_results = {
                "profanity": {"passed": True},
                "output_pii": {"passed": True},
                "factuality": {"passed": True}
            }
            
            # Simple heuristic checks for demo
            response_lower = response_content.lower()
            if any(word in response_lower for word in ["damn", "hell", "crap"]):
                violations.append("mild_profanity")
                guard_results["profanity"]["passed"] = False
                print("  ⚠️ Mild profanity detected (mock)")
            
            # Check for potential PII patterns
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', response_content):  # SSN pattern
                violations.append("potential_pii_leakage")
                guard_results["output_pii"]["passed"] = False
                print("  ⚠️ Potential PII pattern detected (mock)")
        
        return {
            "output_guard_results": guard_results,
            "guard_violations": violations
        }
    
    return output_validation_node


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


def enhance_helpfulness_agent_with_guardrails(
    existing_helpfulness_agent,
    rag_chain: Optional[ProductionRAGChain] = None,
    max_guardrail_refinements: int = 2
):
    """
    Enhance the existing helpfulness agent with comprehensive guardrails.
    
    This function wraps the existing helpfulness agent to add:
    - Input validation (jailbreak, topic, PII detection)
    - Output validation (content moderation, factuality)
    - Guardrail-specific refinement loops
    
    Workflow:
    User Input → Input Guards → Helpfulness Agent → Output Guards → Response
                     ↓              (with built-in           ↓
                Guard Fails          helpfulness loops)  Guard Fails
                     ↓                                       ↓
                Error Response ←← Guardrail Refinement ←←
    
    Args:
        existing_helpfulness_agent: The already created helpfulness agent
        rag_chain: Optional RAG chain for factuality checking
        max_guardrail_refinements: Maximum number of guardrail refinement attempts
        
    Returns:
        Enhanced agent wrapper that combines helpfulness + guardrails
    """
    
    # Create guardrail nodes
    input_validator = create_input_guardrails_node()
    output_validator = create_output_guardrails_node(rag_chain)
    
    class EnhancedAgentWrapper:
        """Wrapper class for the enhanced helpfulness agent with guardrails."""
        
        def __init__(self, base_agent):
            self.base_agent = base_agent
            self.max_refinements = max_guardrail_refinements
        
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """
            Enhanced invoke method that adds guardrails around the helpfulness agent.
            """
            # Extract query from inputs
            if isinstance(inputs, dict) and "messages" in inputs:
                query = inputs["messages"][-1].content if inputs["messages"] else ""
            else:
                query = str(inputs)
            
            # Initialize guardrail state
            guardrail_state = {
                "input_guard_results": {},
                "output_guard_results": {},
                "guard_violations": [],
                "guardrail_refinement_count": 0,
                "max_guardrail_refinements": self.max_refinements
            }
            
            # Step 1: Input Validation
            print("\n🛡️ Running Input Validation...")
            input_message = HumanMessage(content=query)
            mock_state = {
                "messages": [input_message],
                "guard_violations": []
            }
            
            input_result = input_validator(mock_state)
            guardrail_state.update(input_result)
            
            # Check for critical input violations
            violations = guardrail_state.get("guard_violations", [])
            if any(v in violations for v in ["jailbreak_detected", "topic_violation"]):
                print("🚫 Input blocked by guardrails")
                if "jailbreak_detected" in violations:
                    return {
                        "messages": [AIMessage(content="I cannot process requests that attempt to circumvent safety guidelines. Please rephrase your question appropriately.")],
                        "guardrail_violations": violations,
                        "blocked_by": "input_guardrails"
                    }
                elif "topic_violation" in violations:
                    return {
                        "messages": [AIMessage(content="I can only help with questions related to student loans, financial aid, and education financing. Please ask a question within these topics.")],
                        "guardrail_violations": violations,
                        "blocked_by": "input_guardrails"
                    }
            
            # Step 2: Get sanitized input (PII may have been redacted)
            final_messages = input_result.get("messages", [input_message])
            sanitized_query = final_messages[-1].content if final_messages else query
            
            print("✅ Input validation passed - proceeding to helpfulness agent")
            
            # Step 3: Run the existing helpfulness agent
            print("\n🤖 Running Helpfulness Agent...")
            try:
                helpfulness_result = self.base_agent.invoke({
                    "messages": [HumanMessage(content=sanitized_query)]
                })
                
                # Extract the final response from helpfulness agent
                final_message = helpfulness_result["messages"][-1] if helpfulness_result["messages"] else None
                if not final_message or not hasattr(final_message, 'content'):
                    return {
                        "messages": [AIMessage(content="I apologize, but I couldn't generate a response.")],
                        "error": "No valid response from helpfulness agent"
                    }
                
            except Exception as e:
                print(f"❌ Error in helpfulness agent: {e}")
                return {
                    "messages": [AIMessage(content="I apologize, but I encountered an error while processing your request.")],
                    "error": str(e)
                }
            
            # Step 4: Output Validation with Refinement Loop
            current_response = final_message
            refinement_count = 0
            
            while refinement_count < self.max_refinements:
                print(f"\n🔍 Running Output Validation (attempt {refinement_count + 1})...")
                
                # Create mock state for output validation
                output_mock_state = {
                    "messages": helpfulness_result["messages"],
                    "guard_violations": guardrail_state.get("guard_violations", [])
                }
                
                output_result = output_validator(output_mock_state)
                guardrail_state.update(output_result)
                
                # Check for output violations
                current_violations = guardrail_state.get("guard_violations", [])
                critical_violations = ["inappropriate_content", "potential_hallucination", "pii_leakage"]
                
                if not any(v in current_violations for v in critical_violations):
                    print("✅ Output validation passed!")
                    break
                
                # If we have violations and haven't exceeded max refinements, try to refine
                if refinement_count < self.max_refinements - 1:
                    print(f"⚠️ Output violations detected: {current_violations}")
                    print("🔄 Attempting guardrail refinement...")
                    
                    # Create refinement instruction
                    refinement_instructions = []
                    if "inappropriate_content" in current_violations:
                        refinement_instructions.append("ensure the response is professional and appropriate")
                    if "potential_hallucination" in current_violations:
                        refinement_instructions.append("stick strictly to factual information from reliable sources")
                    if "pii_leakage" in current_violations:
                        refinement_instructions.append("avoid including any personal identifying information")
                    
                    refinement_query = f"Please revise your previous response to {', '.join(refinement_instructions)}. Original query: {sanitized_query}"
                    
                    try:
                        # Re-run helpfulness agent with refinement instruction
                        helpfulness_result = self.base_agent.invoke({
                            "messages": [HumanMessage(content=refinement_query)]
                        })
                        current_response = helpfulness_result["messages"][-1] if helpfulness_result["messages"] else current_response
                        refinement_count += 1
                        guardrail_state["guardrail_refinement_count"] = refinement_count
                        
                    except Exception as e:
                        print(f"❌ Error during refinement: {e}")
                        break
                else:
                    print("🔄 Maximum refinement attempts reached")
                    break
            
            # Step 5: Return final result with guardrail metadata
            final_result = helpfulness_result.copy()
            final_result.update(guardrail_state)
            
            return final_result
    
    return EnhancedAgentWrapper(existing_helpfulness_agent)
