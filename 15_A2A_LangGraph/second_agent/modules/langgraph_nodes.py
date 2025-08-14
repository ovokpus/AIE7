# second_agent/modules/langgraph_nodes.py
"""
LangGraph nodes for the Expert Agent System
"""
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.agent_graph_with_helpfulness import AgentState
from .expert_profiles import ExpertProfile, EXPERT_PROFILES
from .logging_config import get_loggers

logger, _ = get_loggers()


class ClientAgentState(AgentState):
    """Extended state for expert agent that can call the A2A server"""
    server_response: Dict[str, Any] = {}
    expert_profile: ExpertProfile = None
    query_type: str = "expert_driven"
    conversation_context: Dict[str, Any] = {}
    synthesis_required: bool = False
    follow_up_needed: bool = False
    interaction_count: int = 0


def create_expert_nodes(a2a_client):
    """Create all the expert agent nodes with the given A2A client"""
    
    async def select_expert(state: ClientAgentState) -> Dict[str, Any]:
        """Select or confirm expert profile for the session"""
        
        logger.info("👨‍🔬 EXPERT SELECTION: Determining expert profile...")
        
        # If no expert profile set, this is initial selection
        if not state.get("expert_profile"):
            logger.info("🎯 No expert profile set - this should be set during session initialization")
            # Default to ML expert for demo purposes
            expert_profile = EXPERT_PROFILES["ml_expert_kimi"]
            logger.info(f"🎭 Using default expert: {expert_profile.name}")
        else:
            expert_profile = state["expert_profile"]
            logger.info(f"🎭 Continuing with expert: {expert_profile.name}")
        
        # Log expert details
        logger.info(f"🧠 Expert Identity: {expert_profile.identity}")
        logger.info(f"🎯 Current Goal: {expert_profile.current_goal}")
        logger.info(f"📊 Quality Standards: {expert_profile.quality_standards}")
        logger.info(f"🔄 Follow-up Strategy: {expert_profile.follow_up_strategy}")
        
        return {
            "expert_profile": expert_profile,
            "query_type": "expert_driven"
        }

    async def call_a2a_agent(state: ClientAgentState) -> Dict[str, Any]:
        """Call the A2A agent with expert profile context"""
        try:
            logger.info("📞 A2A CALL NODE: Preparing expert consultation...")
            
            # Initialize client if not already done
            if not a2a_client.client:
                logger.info("🔗 A2A client not initialized, connecting now...")
                success = await a2a_client.initialize()
                if not success:
                    logger.error("❌ Failed to initialize A2A client")
                    return {
                        "messages": [AIMessage(content="❌ Failed to connect to A2A agent")],
                        "server_response": {"error": "Connection failed"}
                    }
            
            # Get expert profile and query
            expert_profile = state.get("expert_profile")
            if not expert_profile:
                logger.error("❌ No expert profile found in state")
                return {
                    "messages": [AIMessage(content="❌ No expert profile configured")],
                    "server_response": {"error": "No expert profile"}
                }
            
            query = state["messages"][-1].content
            
            # Generate expert persona context
            expert_context = expert_profile.get_persona_context()
            
            logger.info(f"🎯 Expert: {expert_profile.name} ({expert_profile.identity})")
            logger.info(f"🎯 Goal: {expert_profile.current_goal}")
            logger.info(f"🎭 Using expert persona context ({len(expert_context)} chars)")
            
            # Create expert-driven query
            expert_query = f"""As {expert_profile.identity}, I am working on: {expert_profile.current_goal}

                        My question: {query}

                        Please provide a response that meets my standards: {', '.join(expert_profile.quality_standards)}"""

            logger.info(f"📝 Expert query: {expert_query[:150]}...")
            
            # Send message with expert context
            logger.info("🚀 Consulting main agent as expert...")
            expert_profile.questions_asked += 1
            
            result = await a2a_client.send_message_with_context(expert_query, expert_context)
            
            if "error" in result:
                return {
                    "messages": [AIMessage(content=f"❌ Error from A2A agent: {result['error']}")],
                    "server_response": result
                }
            
            # Extract response content - handle the JSONRPC response format
            logger.info("📊 Parsing response from main agent...")
            try:
                if 'result' in result and 'artifacts' in result['result']:
                    # Extract from artifacts
                    artifacts = result['result']['artifacts']
                    if artifacts and len(artifacts) > 0 and 'parts' in artifacts[0]:
                        response_text = artifacts[0]['parts'][0]['text']
                        response_msg = AIMessage(content=response_text)
                        
                        logger.info(f"✅ Successfully parsed response from main agent")
                        logger.info(f"📝 Response length: {len(response_text)} characters")
                        
                        expert_profile = state.get("expert_profile")
                        if expert_profile:
                            logger.info(f"🎭 Expert context used: {expert_profile.name}")
                        
                        # Log first 100 chars of response for demo
                        preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
                        logger.info(f"👀 Response preview: {preview}")
                        
                        return {
                            "messages": [response_msg],
                            "server_response": result
                        }
                
                # Fallback: try the old format
                if 'root' in result and 'result' in result['root']:
                    response_text = result['root']['result']['parts'][0]['text']
                    response_msg = AIMessage(content=response_text)
                    
                    expert_profile = state.get("expert_profile")
                    if expert_profile:
                        logger.info(f"✅ Received response for expert: {expert_profile.name}")
                    else:
                        logger.info(f"✅ Received response from A2A agent")
                    
                    return {
                        "messages": [response_msg],
                        "server_response": result
                    }
                
                # If we can't parse, show the structure
                logger.warning(f"⚠️ Unexpected response format. Keys: {list(result.keys())}")
                return {
                    "messages": [AIMessage(content=f"Received response but couldn't parse format. Keys: {list(result.keys())}")],
                    "server_response": result
                }
                
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"⚠️ Error parsing response: {e}")
                return {
                    "messages": [AIMessage(content=f"Received response but couldn't parse: {str(e)}")],
                    "server_response": result
                }

        except Exception as e:
            logger.error(f"❌ Failed to call A2A agent: {e}")
            return {
                "messages": [AIMessage(content=f"❌ Error calling A2A agent: {e}")],
                "server_response": {"error": str(e)}
            }
    
    async def evaluate_quality(state: ClientAgentState) -> Dict[str, Any]:
        """Evaluate if the response meets the expert's quality standards"""
        logger.info("🔍 QUALITY EVALUATION: Assessing response quality...")
        
        expert_profile = state.get("expert_profile")
        last_message = state["messages"][-1]
        
        if not expert_profile:
            logger.warning("⚠️ No expert profile for quality evaluation")
            return {"follow_up_needed": False}
        
        # Evaluate response quality
        quality_score = expert_profile.evaluate_response_quality(last_message.content)
        expert_profile.satisfaction_level = quality_score
        
        logger.info(f"📊 Quality Score: {quality_score}/10")
        logger.info(f"🎯 Expert Standards: {expert_profile.quality_standards}")
        
        # Determine if follow-up is needed (score < 7 means unsatisfactory)
        follow_up_needed = quality_score < 7 and expert_profile.questions_asked < 3
        
        if follow_up_needed:
            logger.info("❌ Response quality insufficient - follow-up needed")
            logger.info(f"🔄 Follow-up strategy: {expert_profile.follow_up_strategy}")
        else:
            logger.info("✅ Response quality acceptable or max attempts reached")
        
        return {
            "follow_up_needed": follow_up_needed,
            "interaction_count": state.get("interaction_count", 0) + 1
        }
    
    async def generate_follow_up(state: ClientAgentState) -> Dict[str, Any]:
        """Generate a follow-up question based on expert's standards"""
        logger.info("🔄 FOLLOW-UP GENERATION: Creating expert follow-up...")
        
        expert_profile = state.get("expert_profile")
        last_response = state["messages"][-1].content
        
        # Create follow-up based on what's missing
        follow_up_templates = {
            "sources": "I need sources and references to verify this information. Can you provide specific papers, studies, or authoritative sources?",
            "depth": "This seems like a surface-level answer. Can you provide more technical depth and detailed explanations?",
            "specifics": "Can you be more specific? I need concrete examples, numbers, or implementation details.",
            "technical": "Can you provide the technical implementation details, code examples, or mathematical formulations?"
        }
        
        # Determine what type of follow-up is needed
        follow_up_type = "depth"  # default
        if "want sources" in " ".join(expert_profile.quality_standards):
            if not any(word in last_response.lower() for word in ["source", "paper", "study", "http", "arxiv"]):
                follow_up_type = "sources"
        elif "technical" in " ".join(expert_profile.quality_standards):
            follow_up_type = "technical"
        elif len(last_response.split()) < 100:
            follow_up_type = "depth"
        
        follow_up_question = follow_up_templates[follow_up_type]
        
        logger.info(f"🎯 Follow-up type: {follow_up_type}")
        logger.info(f"❓ Follow-up question: {follow_up_question[:100]}...")
        
        # Add follow-up as new message
        follow_up_msg = HumanMessage(content=follow_up_question)
        
        return {
            "messages": [follow_up_msg],
            "follow_up_needed": False  # Reset for next cycle
        }

    async def enhance_response(state: ClientAgentState) -> Dict[str, Any]:
        """Enhance the response with expert metadata and context"""
        logger.info("✨ ENHANCE NODE: Adding expert metadata and formatting...")

        last_message = state["messages"][-1]
        expert_profile = state.get("expert_profile")
        interaction_count = state.get("interaction_count", 1)
        
        if expert_profile:
            logger.info(f"🎨 Enhancing response for expert: {expert_profile.name}")
            logger.info(f"📊 Quality satisfaction: {expert_profile.satisfaction_level}/10")
            logger.info(f"🔄 Interaction #{interaction_count}")
        
        # Add expert metadata to the response
        if expert_profile:
            enhanced_content = f"""**👨‍🔬 Expert Consultation Complete**

**Expert**: {expert_profile.name} ({expert_profile.identity})
**Goal**: {expert_profile.current_goal}
**Quality Score**: {expert_profile.satisfaction_level}/10
**Interaction**: #{interaction_count}

---

{last_message.content}

---
*Expert Standards Met*: {', '.join(expert_profile.quality_standards)}
*Questions Asked*: {expert_profile.questions_asked}
*🕐 Generated at*: {datetime.now().strftime('%H:%M:%S')}"""
        else:
            enhanced_content = f"""**🤖 Response via A2A Agent**

{last_message.content}

---
*🕐 Generated at: {datetime.now().strftime('%H:%M:%S')}*"""
        
        enhanced_msg = AIMessage(content=enhanced_content)
        
        logger.info("✅ Expert response enhancement complete")
        logger.info(f"📊 Final response length: {len(enhanced_content)} characters")
        
        return {"messages": [enhanced_msg]}

    def should_continue(state: ClientAgentState) -> str:
        """Route through the expert agent workflow"""
        if not state.get("expert_profile"):
            return "select_expert"
        elif not state.get("server_response"):
            return "call_a2a"
        elif state.get("follow_up_needed"):
            return "generate_follow_up"
        elif "follow_up_needed" not in state:  # Need to evaluate quality
            return "evaluate_quality"
        else:
            return "enhance"
    
    return {
        "select_expert": select_expert,
        "call_a2a_agent": call_a2a_agent,
        "evaluate_quality": evaluate_quality,
        "generate_follow_up": generate_follow_up,
        "enhance_response": enhance_response,
        "should_continue": should_continue
    }


def create_expert_agent(a2a_client):
    """Create the expert agent LangGraph with all nodes"""
    
    # Get all the nodes
    nodes = create_expert_nodes(a2a_client)
    
    # Build the expert agent graph
    graph = StateGraph(ClientAgentState)

    graph.add_node("select_expert", nodes["select_expert"])
    graph.add_node("call_a2a", nodes["call_a2a_agent"])
    graph.add_node("evaluate_quality", nodes["evaluate_quality"])
    graph.add_node("generate_follow_up", nodes["generate_follow_up"])
    graph.add_node("enhance", nodes["enhance_response"])

    # Set up the flow with conditional routing
    graph.set_entry_point("select_expert")
    graph.add_edge("select_expert", "call_a2a")
    graph.add_edge("call_a2a", "evaluate_quality")
    
    # Conditional edges for follow-up logic
    graph.add_conditional_edges(
        "evaluate_quality",
        nodes["should_continue"],
        {
            "generate_follow_up": "generate_follow_up",
            "enhance": "enhance"
        }
    )
    
    # Follow-up loops back to call A2A
    graph.add_edge("generate_follow_up", "call_a2a")
    
    # End after enhancement
    graph.add_edge("enhance", END)
    
    return graph.compile()
