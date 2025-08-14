# second_agent/main.py - LangGraph Client for A2A Communication
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from a2a.client import A2ACardResolver, A2AClient as A2AProtocolClient
from a2a.types import MessageSendParams, SendMessageRequest

# Import from your app
from app.agent_graph_with_helpfulness import AgentState

load_dotenv()

# Enhanced logging configuration
class ColorFormatter(logging.Formatter):
    """Colored logging formatter for better visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[94m{record.name}\033[0m"  # Blue
        return super().format(record)

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Apply color formatter to console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))

# Create logger
logger = logging.getLogger("SecondAgent")
logger.handlers.clear()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Create a separate interaction logger
interaction_logger = logging.getLogger("Interaction")
interaction_logger.handlers.clear()
interaction_logger.addHandler(console_handler)
interaction_logger.setLevel(logging.INFO)


class ExpertProfile:
    """Represents a specific expert agent with goals and standards"""
    def __init__(self, 
                 name: str,
                 identity: str, 
                 current_goal: str, 
                 quality_standards: List[str],
                 follow_up_strategy: str,
                 domain_expertise: List[str]):
        self.name = name
        self.identity = identity
        self.current_goal = current_goal
        self.quality_standards = quality_standards
        self.follow_up_strategy = follow_up_strategy
        self.domain_expertise = domain_expertise
        self.questions_asked = 0
        self.satisfaction_level = 0  # 0-10 scale
        
    def get_persona_context(self) -> str:
        """Generate persona context for A2A calls"""
        standards_text = " and ".join(self.quality_standards)
        return f"""You are {self.identity}. Your current goal is: {self.current_goal}.
        
                    Quality Standards: {standards_text}
                            
                    Domain Expertise: {', '.join(self.domain_expertise)}

                    Follow-up Strategy: {self.follow_up_strategy}

                    Remember: You are a persistent expert who won't settle for superficial answers. Ask follow-up questions if needed."""

    def evaluate_response_quality(self, response: str) -> int:
        """Evaluate if response meets quality standards (0-10)"""
        score = 5  # baseline
        
        # Check for depth indicators
        if len(response) > 500:
            score += 1
        if "research" in response.lower() or "study" in response.lower():
            score += 1
        if "source" in response.lower() or "paper" in response.lower():
            score += 1
        if any(expertise.lower() in response.lower() for expertise in self.domain_expertise):
            score += 1
            
        # Check against quality standards
        if "not satisfied with surface level" in " ".join(self.quality_standards):
            if len(response.split()) < 100:  # Too short
                score -= 2
            if "detailed" in response.lower() or "technical" in response.lower():
                score += 1
                
        if "want sources" in " ".join(self.quality_standards):
            if "http" in response or "arxiv" in response.lower() or "doi" in response.lower():
                score += 2
            else:
                score -= 1
                
        return min(10, max(0, score))


class ClientAgentState(AgentState):
    """Extended state for expert agent that can call the A2A server"""
    server_response: Dict[str, Any] = {}
    expert_profile: ExpertProfile = None
    query_type: str = "expert_driven"
    conversation_context: Dict[str, Any] = {}
    synthesis_required: bool = False
    follow_up_needed: bool = False
    interaction_count: int = 0


class A2AClientWrapper:
    """Client for communicating with the local A2A agent server"""
    
    def __init__(self, base_url: str = 'http://localhost:10000'):
        self.base_url = base_url
        self.client = None
        self.agent_card = None

    async def initialize(self):
        """Initialize the A2A client connection"""
        try:
            logger.info("🔗 Initializing A2A client connection...")
            logger.info(f"📡 Connecting to: {self.base_url}")
            
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url
            )
            
            logger.info("📋 Fetching agent card...")
            self.agent_card = await resolver.get_agent_card()
            
            logger.info(f"🤖 Main agent capabilities: {self.agent_card.skills}")
            
            self.client = A2AProtocolClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
            logger.info(f"✅ A2A Client initialized for '{self.agent_card.name}'")
            logger.info(f"🎯 Available skills: {[skill.name for skill in self.agent_card.skills]}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize A2A client: {e}")
            return False

    async def send_message_with_context(self, message: str, persona_context: str = None) -> Dict[str, Any]:
        """Send a message to the A2A agent with optional persona context"""
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        # Add persona context if provided
        full_message = message
        if persona_context:
            full_message = f"Context: {persona_context}\n\nQuery: {message}"
            logger.info(f"🎭 Adding persona context: {persona_context[:50]}...")
        
        logger.info(f"📝 Original query: {message}")
        logger.info(f"📤 Sending enhanced message to main agent...")
        
        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': full_message}],
                'message_id': uuid4().hex,
            }
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )
        
        try:
            logger.info("⏳ Waiting for main agent response...")
            start_time = time.time()
            
            response = await self.client.send_message(request)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"📥 Received response from main agent ({response_time:.2f}s)")
            
            return response.model_dump(mode='json', exclude_none=True)
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return {"error": str(e)}


def create_expert_agent():
    """Create an expert agent with specific goals and quality standards"""
    
    # Initialize A2A client
    a2a_client = A2AClientWrapper()
    
    # Define specific expert profiles matching assignment requirements
    EXPERT_PROFILES = {
        "ml_expert_kimi": ExpertProfile(
            name="Dr. Sarah Chen",
            identity="an expert in Machine Learning",
            current_goal="learn about what makes Kimi K2 so incredible",
            quality_standards=[
                "not satisfied with surface level answers",
                "want sources to read to verify information"
            ],
            follow_up_strategy="If initial answer lacks depth, ask for technical details, papers, or implementation specifics",
            domain_expertise=["machine learning", "neural networks", "language models", "AI architectures"]
        ),
        
        "ai_researcher_transformers": ExpertProfile(
            name="Prof. Marcus Rodriguez",
            identity="an AI researcher specializing in transformer architectures",
            current_goal="understand the latest innovations in attention mechanisms and their practical applications",
            quality_standards=[
                "need academic rigor and citations",
                "require technical implementation details"
            ],
            follow_up_strategy="Demand mathematical explanations and code examples when concepts are mentioned",
            domain_expertise=["transformers", "attention mechanisms", "deep learning", "NLP"]
        ),
        
        "startup_founder_ai": ExpertProfile(
            name="Alex Kim",
            identity="a startup founder building an AI-powered product",
            current_goal="evaluate which AI technologies to build on and understand their business implications",
            quality_standards=[
                "need practical implementation details",
                "require cost analysis and ROI data"
            ],
            follow_up_strategy="Ask for real-world examples, pricing, and scalability concerns",
            domain_expertise=["AI APIs", "business strategy", "product development", "scaling AI"]
        ),
        
        "security_expert_ai": ExpertProfile(
            name="Dr. Emma Watson",
            identity="a cybersecurity expert investigating AI system vulnerabilities",
            current_goal="understand security risks in large language models and mitigation strategies",
            quality_standards=[
                "need concrete examples of vulnerabilities",
                "require mitigation strategies with evidence"
            ],
            follow_up_strategy="Ask for specific attack vectors and defense mechanisms with technical proof",
            domain_expertise=["AI security", "prompt injection", "model safety", "adversarial attacks"]
        )
    }

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

    # Build the expert agent graph
    graph = StateGraph(ClientAgentState)

    graph.add_node("select_expert", select_expert)
    graph.add_node("call_a2a", call_a2a_agent)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("generate_follow_up", generate_follow_up)
    graph.add_node("enhance", enhance_response)

    # Set up the flow with conditional routing
    graph.set_entry_point("select_expert")
    graph.add_edge("select_expert", "call_a2a")
    graph.add_edge("call_a2a", "evaluate_quality")
    
    # Conditional edges for follow-up logic
    graph.add_conditional_edges(
        "evaluate_quality",
        should_continue,
        {
            "generate_follow_up": "generate_follow_up",
            "enhance": "enhance"
        }
    )
    
    graph.add_edge("generate_follow_up", "call_a2a")  # Loop back for follow-up
    graph.add_edge("enhance", END)

    return graph.compile()

async def demo_single_query():
    """Demo with a single query to show basic functionality"""
    print("🤖 Single Query Demo")
    print("=" * 40)
    
    expert_agent = create_expert_agent()
    
    # Use default ML expert for demo
    ml_expert = ExpertProfile(
        name="Dr. Sarah Chen",
        identity="an expert in Machine Learning",
        current_goal="learn about what makes Kimi K2 so incredible",
        quality_standards=[
            "not satisfied with surface level answers",
            "want sources to read to verify information"
        ],
        follow_up_strategy="If initial answer lacks depth, ask for technical details, papers, or implementation specifics",
        domain_expertise=["machine learning", "neural networks", "language models", "AI architectures"]
    )

    initial_state = {
        "messages": [HumanMessage(content="What makes Kimi K2 so incredible?")],
        "server_response": {},
        "expert_profile": ml_expert,
        "query_type": "expert_driven",
        "conversation_context": {},
        "synthesis_required": False,
        "follow_up_needed": False,
        "interaction_count": 0
    }
    
    try:
        result = await expert_agent.ainvoke(initial_state)
        print("\n📝 Expert Agent Result:")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_different_query_types():
    """Demo with different types of queries to show expert routing"""
    print("🎭 Multi-Expert Demo")
    print("=" * 40)
    
    expert_agent = create_expert_agent()
    
    # Create different experts for demo
    experts = [
        ExpertProfile(
            name="Dr. Sarah Chen",
            identity="an expert in Machine Learning",
            current_goal="learn about what makes Kimi K2 so incredible",
            quality_standards=[
                "not satisfied with surface level answers",
                "want sources to read to verify information"
            ],
            follow_up_strategy="If initial answer lacks depth, ask for technical details, papers, or implementation specifics",
            domain_expertise=["machine learning", "neural networks", "language models", "AI architectures"]
        ),
        ExpertProfile(
            name="Alex Kim",
            identity="a startup founder building an AI-powered product",
            current_goal="evaluate which AI technologies to build on and understand their business implications",
            quality_standards=[
                "need practical implementation details",
                "require cost analysis and ROI data"
            ],
            follow_up_strategy="Ask for real-world examples, pricing, and scalability concerns",
            domain_expertise=["AI APIs", "business strategy", "product development", "scaling AI"]
        )
    ]
    
    queries = [
        ("ML Expert - Kimi K2", "What makes Kimi K2 so incredible?", experts[0]),
        ("Startup Founder - LLM Costs", "What are the implementation costs and ROI for deploying large language models in enterprise?", experts[1])
    ]
    
    for query_name, query_text, expert in queries:
        print(f"\n{'='*50}")
        print(f"👨‍🔬 {query_name}")
        print(f"🎯 Expert: {expert.name} ({expert.identity})")
        print(f"{'='*50}")
        
        initial_state = {
            "messages": [HumanMessage(content=query_text)],
            "server_response": {},
            "expert_profile": expert,
            "query_type": "expert_driven",
            "conversation_context": {},
            "synthesis_required": False,
            "follow_up_needed": False,
            "interaction_count": 0
        }
        
        try:
            result = await expert_agent.ainvoke(initial_state)
            print(f"\n📝 Expert Response:")
            response_content = result["messages"][-1].content
            # Show first 300 characters
            print(response_content[:300] + "..." if len(response_content) > 300 else response_content)
            print(f"📊 Expert satisfaction: {expert.satisfaction_level}/10")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(1)  # Small delay between queries


async def interactive_mode():
    """Enhanced interactive mode with expert selection"""
    print("\n" + "="*80)
    print("👨‍🔬 EXPERT AGENT - INTERACTIVE MODE")
    print("="*80)
    print("🎯 Experience goal-oriented expert agents with specific research missions!")
    print("📊 Watch experts evaluate responses and ask follow-up questions")
    print("💡 Commands: 'quit'/'exit'/'q' to stop, 'help' for examples, 'switch' to change expert")
    print("="*80)
    
    # Initialize the expert agent
    interaction_logger.info("🚀 Starting expert agent session...")
    expert_agent = create_expert_agent()
    
    # Let user select an expert
    current_expert = await select_expert_profile()
    if not current_expert:
        return
    
    session_count = 0
    
    print(f"\n🎭 **Current Expert**: {current_expert.name}")
    print(f"🧠 **Identity**: {current_expert.identity}")
    print(f"🎯 **Goal**: {current_expert.current_goal}")
    print(f"📊 **Standards**: {', '.join(current_expert.quality_standards)}")

    # Main interactive loop
    while True:
        try:
            print(f"\n{'─'*60}")
            user_input = input(f"🤔 {current_expert.name} asks: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                interaction_logger.info("👋 User ended expert session")
                break
            
            if user_input.lower() == 'switch':
                new_expert = await select_expert_profile()
                if new_expert:
                    current_expert = new_expert
                    print(f"\n🎭 **Switched to Expert**: {current_expert.name}")
                    print(f"🧠 **Identity**: {current_expert.identity}")
                    print(f"🎯 **Goal**: {current_expert.current_goal}")
                continue
            
            if user_input.lower() == 'help':
                print("\n📚 Example questions for your expert:")
                if current_expert.name == "Dr. Sarah Chen":
                    print("🔬 'What makes Kimi K2 so incredible?'")
                    print("🔬 'Tell me about Kimi K2's architecture and capabilities'")
                elif current_expert.name == "Prof. Marcus Rodriguez":
                    print("🧠 'What are the latest innovations in attention mechanisms?'")
                    print("🧠 'How do transformer architectures work?'")
                elif current_expert.name == "Alex Kim":
                    print("💼 'Should I build on LangChain or LlamaIndex?'")
                    print("💼 'What are the costs of deploying LLMs?'")
                elif current_expert.name == "Dr. Emma Watson":
                    print("🔒 'What are the security risks in large language models?'")
                    print("🔒 'How can I protect against prompt injection attacks?'")
                continue
            
            if not user_input:
                continue
            
            session_count += 1
            interaction_logger.info(f"📝 SESSION {session_count}: Expert query from {current_expert.name}")
            interaction_logger.info(f"❓ Query: '{user_input}'")
            
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "server_response": {},
                "expert_profile": current_expert,
                "query_type": "expert_driven",
                "conversation_context": {},
                "synthesis_required": False,
                "follow_up_needed": False,
                "interaction_count": 0
            }
            
            print(f"\n🔄 {current_expert.name} is analyzing your question...")
            print("📊 Watch the expert workflow and quality evaluation below:")
            print("─" * 60)
            
            # Execute the expert workflow with full logging
            start_time = time.time()
            result = await expert_agent.ainvoke(initial_state)
            end_time = time.time()
            
            print("─" * 60)
            print(f"⏱️  Total consultation time: {end_time - start_time:.2f} seconds")
            print(f"📊 Expert satisfaction: {current_expert.satisfaction_level}/10")
            print(f"🔄 Questions asked: {current_expert.questions_asked}")
            print("\n🎯 EXPERT CONSULTATION RESULT:")
            print("="*60)
            print(result["messages"][-1].content)
            print("="*60)
            
            interaction_logger.info(f"✅ SESSION {session_count} completed - Expert: {current_expert.name}")
            
        except KeyboardInterrupt:
            interaction_logger.info("⚠️ Expert session interrupted by user")
            break
        except Exception as e:
            interaction_logger.error(f"❌ Error in expert session {session_count}: {e}")
            print(f"❌ Error: {e}")
    
    print(f"\n👋 Expert consultation ended after {session_count} queries")
    print(f"🎭 Final expert: {current_expert.name}")
    print(f"📊 Final satisfaction: {current_expert.satisfaction_level}/10")
    print("Thanks for trying the Expert Agent system!")


async def select_expert_profile():
    """Allow user to select which expert profile to use"""
    
    # Get expert profiles (we need to recreate this since it's inside the function)
    expert_profiles = {
        "ml_expert_kimi": ExpertProfile(
            name="Dr. Sarah Chen",
            identity="an expert in Machine Learning",
            current_goal="learn about what makes Kimi K2 so incredible",
            quality_standards=[
                "not satisfied with surface level answers",
                "want sources to read to verify information"
            ],
            follow_up_strategy="If initial answer lacks depth, ask for technical details, papers, or implementation specifics",
            domain_expertise=["machine learning", "neural networks", "language models", "AI architectures"]
        ),
        
        "ai_researcher_transformers": ExpertProfile(
            name="Prof. Marcus Rodriguez",
            identity="an AI researcher specializing in transformer architectures",
            current_goal="understand the latest innovations in attention mechanisms and their practical applications",
            quality_standards=[
                "need academic rigor and citations",
                "require technical implementation details"
            ],
            follow_up_strategy="Demand mathematical explanations and code examples when concepts are mentioned",
            domain_expertise=["transformers", "attention mechanisms", "deep learning", "NLP"]
        ),
        
        "startup_founder_ai": ExpertProfile(
            name="Alex Kim",
            identity="a startup founder building an AI-powered product",
            current_goal="evaluate which AI technologies to build on and understand their business implications",
            quality_standards=[
                "need practical implementation details",
                "require cost analysis and ROI data"
            ],
            follow_up_strategy="Ask for real-world examples, pricing, and scalability concerns",
            domain_expertise=["AI APIs", "business strategy", "product development", "scaling AI"]
        ),
        
        "security_expert_ai": ExpertProfile(
            name="Dr. Emma Watson",
            identity="a cybersecurity expert investigating AI system vulnerabilities",
            current_goal="understand security risks in large language models and mitigation strategies",
            quality_standards=[
                "need concrete examples of vulnerabilities",
                "require mitigation strategies with evidence"
            ],
            follow_up_strategy="Ask for specific attack vectors and defense mechanisms with technical proof",
            domain_expertise=["AI security", "prompt injection", "model safety", "adversarial attacks"]
        )
    }
    
    print("\n🎭 **Available Expert Profiles**:")
    print("1. 🔬 Dr. Sarah Chen - ML Expert studying Kimi K2 (Assignment Example)")
    print("2. 🧠 Prof. Marcus Rodriguez - Transformer Architecture Researcher") 
    print("3. 💼 Alex Kim - AI Startup Founder")
    print("4. 🔒 Dr. Emma Watson - AI Security Expert")
    
    try:
        choice = input("\nSelect expert (1-4): ").strip()
        expert_map = {
            "1": "ml_expert_kimi",
            "2": "ai_researcher_transformers", 
            "3": "startup_founder_ai",
            "4": "security_expert_ai"
        }
        
        if choice in expert_map:
            selected_expert = expert_profiles[expert_map[choice]]
            interaction_logger.info(f"👨‍🔬 Expert selected: {selected_expert.name}")
            return selected_expert
        else:
            print("Invalid choice. Using default ML expert...")
            return expert_profiles["ml_expert_kimi"]
            
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return None


async def check_server_status():
    """Check if the A2A server is running"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get("http://localhost:10000/.well-known/agent-card.json")
            if response.status_code == 200:
                print("✅ A2A agent server is running")
                return True
            else:
                print(f"❌ A2A server responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to A2A server: {e}")
        print("💡 Please start your A2A agent with: uv run python -m app")
        return False


async def main():
    """Main demo function with expert agent system"""
    print("🎓 EXPERT AGENT SYSTEM - Advanced A2A Communication")
    print("=" * 60)
    print("🎯 Goal-oriented experts with specific research missions!")
    print("📊 Featuring: Dr. Sarah Chen studying Kimi K2 (Assignment Example)")
    
    # Check server status first
    if not await check_server_status():
        return
    
    print("\nChoose demo type:")
    print("1. Single Query Demo (quick test)")
    print("2. Multi-Persona Demo (original system)")
    print("3. 🌟 Expert Agent Mode (NEW - Assignment Compliant)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        await demo_single_query()
    elif choice == "2":
        await demo_different_query_types()
    elif choice == "3":
        await interactive_mode()
    else:
        print("Invalid choice. Starting Expert Agent Mode...")
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
