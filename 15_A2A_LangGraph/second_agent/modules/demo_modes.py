# second_agent/modules/demo_modes.py
"""
Demo modes for the Expert Agent System
"""
import asyncio
import time

from langchain_core.messages import HumanMessage

from .expert_profiles import ExpertProfile, EXPERT_PROFILES, get_expert_profile
from .langgraph_nodes import create_expert_agent
from .logging_config import get_loggers

logger, interaction_logger = get_loggers()


async def demo_single_query(a2a_client):
    """Demo with a single query to show basic functionality"""
    print("🤖 Single Query Demo")
    print("=" * 40)
    
    expert_agent = create_expert_agent(a2a_client)
    
    # Use default ML expert for demo
    ml_expert = get_expert_profile("ml_expert_kimi")

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


async def demo_different_query_types(a2a_client):
    """Demo with different types of queries to show expert routing"""
    print("🎭 Multi-Expert Demo")
    print("=" * 40)
    
    expert_agent = create_expert_agent(a2a_client)
    
    # Create different experts for demo
    experts = [
        get_expert_profile("ml_expert_kimi"),
        get_expert_profile("startup_founder_ai")
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


async def select_expert_profile():
    """Allow user to select which expert profile to use"""
    
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
            selected_expert = get_expert_profile(expert_map[choice])
            interaction_logger.info(f"👨‍🔬 Expert selected: {selected_expert.name}")
            return selected_expert
        else:
            print("Invalid choice. Using default ML expert...")
            return get_expert_profile("ml_expert_kimi")
            
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return None


async def interactive_mode(a2a_client):
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
    expert_agent = create_expert_agent(a2a_client)
    
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
